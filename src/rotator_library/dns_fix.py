# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
DNS fix module for bypassing system DNS hijacking.

This module provides a monkey-patch for socket.getaddrinfo to use custom DNS
servers for specific hosts. This fixes issues where VPN/proxy/antivirus
hijack DNS and return wrong IPs (e.g., 198.18.0.x instead of real Azure IPs).

Usage:
    Import this module BEFORE importing litellm/aiohttp:
    
    ```python
    from rotator_library.dns_fix import apply_dns_fix
    apply_dns_fix()
    ```
    
    Or set environment variable:
    
    ```bash
    HTTP_DNS_RESOLVER=8.8.8.8  # Use Google DNS
    ```
"""

import os
import socket
import struct
import random
import json
import ssl
import time
import asyncio
from typing import List, Tuple, Optional, Dict

# Try to use httpx for DoH, fallback to urllib
try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
    import urllib.request
    import urllib.error

# Load .env file if available (before reading environment variables)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# Original getaddrinfo function
_original_getaddrinfo = socket.getaddrinfo

# =============================================================================
# DNS CACHE WITH TTL
# =============================================================================
# Cache structure: hostname -> (list of IPs, expiry timestamp)
_dns_cache: Dict[str, Tuple[List[str], float]] = {}


def get_dns_cache_ttl() -> int:
    """Get DNS cache TTL from config."""
    try:
        from .config.defaults import DNS_CACHE_TTL
        return DNS_CACHE_TTL
    except ImportError:
        return 300  # 5 minutes fallback


def _get_cached_ips(hostname: str) -> Optional[List[str]]:
    """Get cached IPs if still valid."""
    if hostname in _dns_cache:
        ips, expiry = _dns_cache[hostname]
        if time.time() < expiry:
            return ips
        else:
            # Expired, remove from cache
            del _dns_cache[hostname]
    return None


def _cache_ips(hostname: str, ips: List[str]) -> None:
    """Cache IPs with TTL."""
    if ips:
        ttl = get_dns_cache_ttl()
        _dns_cache[hostname] = (ips, time.time() + ttl)


def get_dns_query_timeout() -> int:
    """Get DNS query timeout from config."""
    try:
        from .config.defaults import DNS_QUERY_TIMEOUT
        return DNS_QUERY_TIMEOUT
    except ImportError:
        return 10  # 10 seconds fallback


async def resolve_dns_async(hostname: str) -> List[str]:
    """
    Async DNS resolution with caching.
    
    Args:
        hostname: Hostname to resolve
        
    Returns:
        List of IP addresses
        
    Raises:
        Exception: If DNS resolution fails and no cached value available
    """
    # Check cache first
    cached = _get_cached_ips(hostname)
    if cached is not None:
        return cached
    
    # Use asyncio executor for blocking getaddrinfo
    loop = asyncio.get_event_loop()
    timeout = get_dns_query_timeout()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: socket.getaddrinfo(hostname, None)),
            timeout=timeout
        )
        ips = list(set(r[4][0] for r in result if r[0] == socket.AF_INET))
        if ips:
            _cache_ips(hostname, ips)
        return ips
    except Exception:
        # Fallback to cached value even if expired
        if hostname in _dns_cache:
            return _dns_cache[hostname][0]
        raise


def clear_dns_cache() -> None:
    """Clear all cached DNS entries."""
    global _dns_cache
    _dns_cache = {}


def get_dns_cache_stats() -> Dict[str, any]:
    """Get DNS cache statistics."""
    now = time.time()
    valid_entries = sum(1 for ips, expiry in _dns_cache.values() if now < expiry)
    expired_entries = len(_dns_cache) - valid_entries
    return {
        "total_entries": len(_dns_cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "hostnames": list(_dns_cache.keys())
    }


# List of hosts that should use custom DNS
# Format: hostname -> IP address (if known) or None (to use DNS resolver)
CUSTOM_DNS_HOSTS = {
    "noobrouterproduction.azurewebsites.net": None,  # Use DNS resolver (HTTP_DNS_RESOLVER) instead of hardcoded IP
    # Add more hosts as needed
}


def _get_doh_timeout() -> int:
    """Get DoH query timeout from config."""
    try:
        from .config.defaults import HTTP_DOH_TIMEOUT
        return HTTP_DOH_TIMEOUT
    except ImportError:
        return 5  # 5 seconds fallback


def _doh_query(host: str, doh_url: str) -> Optional[str]:
    """
    Query DNS over HTTPS (DoH) for A record.
    
    Args:
        host: Hostname to resolve
        doh_url: DoH endpoint URL (e.g., "https://cloudflare-dns.com/dns-query")
    
    Returns:
        IP address string or None if failed
    """
    try:
        url = f"{doh_url}?name={host}&type=A"
        headers = {"Accept": "application/dns-json"}
        
        if _HAS_HTTPX:
            with _httpx.Client(verify=True, timeout=_get_doh_timeout()) as client:
                response = client.get(url, headers=headers)
                data = response.json()
        else:
            req = urllib.request.Request(url, headers=headers)
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=_get_doh_timeout(), context=ctx) as resp:
                data = json.loads(resp.read().decode())
        
        if data.get("Status") == 0 and "Answer" in data:
            for answer in data["Answer"]:
                if answer.get("type") == 1:  # Type A
                    return answer["data"]
        
        return None
        
    except Exception as e:
        print(f"[DNS-FIX] DoH query error: {e}")
        return None


def _dns_query(host: str, dns_host: str, dns_port: int = 53) -> Optional[str]:
    """
    Query DNS server for A record.
    
    Args:
        host: Hostname to resolve
        dns_host: DNS server IP
        dns_port: DNS server port (default: 53)
    
    Returns:
        IP address string or None if failed
    """
    try:
        # Create DNS query for A record
        query_id = random.randint(0, 65535)
        
        # Header: ID, Flags (standard query), QDCOUNT=1, ANCOUNT=0, NSCOUNT=0, ARCOUNT=0
        query = struct.pack("!HHHHHH", query_id, 0x0100, 1, 0, 0, 0)
        
        # Question: domain name (encode each label with length prefix)
        for part in host.split('.'):
            query += bytes([len(part)]) + part.encode('ascii')
        query += b'\x00'  # Null terminator
        
        # Question: type A (1), class IN (1)
        query += struct.pack("!HH", 1, 1)
        
        # Send DNS query via UDP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.sendto(query, (dns_host, dns_port))
        
        # Receive response
        response, _ = sock.recvfrom(512)
        sock.close()
        
        # Parse response
        # Skip header (12 bytes)
        offset = 12
        
        # Skip question section
        while response[offset] != 0:
            offset += response[offset] + 1
        offset += 5  # Skip null byte and QTYPE/QCLASS
        
        # Parse answer section
        # Skip name (could be pointer)
        if response[offset] & 0xC0 == 0xC0:
            offset += 2  # Pointer
        else:
            while response[offset] != 0:
                offset += response[offset] + 1
            offset += 1
        
        # Parse TYPE, CLASS, TTL, RDLENGTH
        rtype, rclass, ttl, rdlength = struct.unpack("!HHIH", response[offset:offset+10])
        offset += 10
        
        if rtype == 1 and rdlength == 4:  # Type A, IPv4
            ip_bytes = response[offset:offset+4]
            ip = ".".join(str(b) for b in ip_bytes)
            return ip
        else:
            return None
        
    except Exception as e:
        print(f"[DNS-FIX] Error querying DNS: {e}")
        return None


def _custom_getaddrinfo(host: str, port: int, family: int = 0, type: int = 0, proto: int = 0, flags: int = 0) -> List[Tuple]:
    """
    Custom getaddrinfo that uses custom DNS for specific hosts.
    
    Args:
        host: Hostname to resolve
        port: Port number
        family: Address family filter
        type: Socket type filter
        proto: Protocol filter
        flags: Flags filter
    
    Returns:
        List of address tuples
    """
    if host in CUSTOM_DNS_HOSTS:
        # Check if we have a known IP for this host
        known_ip = CUSTOM_DNS_HOSTS[host]
        if known_ip:
            print(f"[DNS-FIX] Using known IP for {host} -> {known_ip}")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (known_ip, port))]
        
        # Otherwise, use DNS resolver
        dns_resolver = os.getenv("HTTP_DNS_RESOLVER", "").strip()
        
        if dns_resolver and dns_resolver.lower() not in ("false", "0", "no", "off"):
            ip = None
            
            # Check if it's a DoH URL
            if dns_resolver.startswith("http://") or dns_resolver.startswith("https://"):
                # Use DoH
                ip = _doh_query(host, dns_resolver)
                if ip:
                    print(f"[DNS-FIX] Resolved {host} -> {ip} via DoH: {dns_resolver}")
            else:
                # Use traditional DNS
                dns_host = dns_resolver
                dns_port = 53
                
                if ":" in dns_resolver:
                    parts = dns_resolver.rsplit(":", 1)
                    dns_host = parts[0]
                    try:
                        dns_port = int(parts[1])
                    except ValueError:
                        pass
                
                ip = _dns_query(host, dns_host, dns_port)
                if ip:
                    print(f"[DNS-FIX] Resolved {host} -> {ip} via {dns_host}:{dns_port}")
            
            if ip:
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]
            else:
                print(f"[DNS-FIX] Failed to resolve {host} via custom DNS, falling back to system DNS")
    
    # Use system DNS for other hosts or if custom DNS failed
    return _original_getaddrinfo(host, port, family, type, proto, flags)


def apply_dns_fix():
    """
    Apply DNS fix by monkey-patching socket.getaddrinfo.
    
    This should be called BEFORE importing litellm/aiohttp.
    """
    dns_resolver = os.getenv("HTTP_DNS_RESOLVER", "").strip()
    
    # Check if custom DNS is disabled
    if not dns_resolver or dns_resolver.lower() in ("false", "0", "no", "off"):
        return
    
    # Parse DNS resolver
    dns_host = dns_resolver
    dns_port = 53
    
    if ":" in dns_resolver:
        parts = dns_resolver.rsplit(":", 1)
        dns_host = parts[0]
        try:
            dns_port = int(parts[1])
        except ValueError:
            pass
    
    # Apply monkey-patch
    socket.getaddrinfo = _custom_getaddrinfo
    print(f"[DNS-FIX] Patched socket.getaddrinfo to use custom DNS: {dns_host}:{dns_port}")


# Auto-apply when module is imported
apply_dns_fix()
