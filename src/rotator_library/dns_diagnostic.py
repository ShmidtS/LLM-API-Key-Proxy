#!/usr/bin/env python3
"""
DNS Diagnostic Script for chatgpt.com

This script helps diagnose DNS and connection issues by:
1. Showing which IP is returned from CUSTOM_DNS_HOSTS
2. Showing which IP is returned from system DNS resolver
3. Testing connection with different IPs
4. Showing Host header behavior
5. Showing server response (status code, headers, body)
"""

import socket
import ssl
import sys
from typing import Optional, Dict, Any


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def get_custom_dns_ip() -> Optional[str]:
    """Get IP from CUSTOM_DNS_HOSTS in dns_fix.py."""
    try:
        from rotator_library.dns_fix import CUSTOM_DNS_HOSTS

        host = "chatgpt.com"
        ip = CUSTOM_DNS_HOSTS.get(host)
        print(f"CUSTOM_DNS_HOSTS['{host}'] = {ip}")
        return ip
    except ImportError as e:
        print(f"Could not import CUSTOM_DNS_HOSTS: {e}")
        return None


def get_system_dns_ip() -> Optional[str]:
    """Get IP from system DNS resolver."""
    host = "chatgpt.com"
    try:
        ip = socket.gethostbyname(host)
        print(f"System DNS resolved: {host} -> {ip}")
        return ip
    except socket.gaierror as e:
        print(f"System DNS resolution failed: {e}")
        return None


def get_google_dns_ip() -> Optional[str]:
    """Get IP using Google DNS (8.8.8.8)."""
    host = "chatgpt.com"
    try:
        # Use socket with custom DNS
        import struct
        import random

        # Create DNS query
        query_id = random.randint(0, 65535)
        query = struct.pack("!HHHHHH", query_id, 0x0100, 1, 0, 0, 0)

        for part in host.split("."):
            query += bytes([len(part)]) + part.encode("ascii")
        query += b"\x00"
        query += struct.pack("!HH", 1, 1)  # Type A, Class IN

        # Send to Google DNS
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.sendto(query, ("8.8.8.8", 53))

        response, _ = sock.recvfrom(512)
        sock.close()

        # Parse response
        offset = 12
        while response[offset] != 0:
            offset += response[offset] + 1
        offset += 5

        if response[offset] & 0xC0 == 0xC0:
            offset += 2
        else:
            while response[offset] != 0:
                offset += response[offset] + 1
            offset += 1

        offset += 4  # Skip TYPE and CLASS
        offset += 4  # Skip TTL
        rdlength = struct.unpack("!H", response[offset : offset + 2])[0]
        offset += 2

        if rdlength == 4:
            ip = ".".join(str(b) for b in response[offset : offset + 4])
            print(f"Google DNS (8.8.8.8) resolved: {host} -> {ip}")
            return ip
        else:
            print(f"Unexpected RDLENGTH: {rdlength}")
            return None

    except Exception as e:
        print(f"Google DNS resolution failed: {e}")
        return None


def test_connection(ip: str, host: str, use_ssl: bool = True) -> Dict[str, Any]:
    """Test connection to the given IP."""
    result = {
        "ip": ip,
        "host": host,
        "success": False,
        "status_code": None,
        "headers": {},
        "body": "",
        "error": None,
    }

    try:
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)

        # Connect
        sock.connect((ip, 443))

        if use_ssl:
            # Wrap with SSL
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            sock = context.wrap_socket(sock, server_hostname=host)

        # Send HTTP request
        request = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"User-Agent: DNS-Diagnostic/1.0\r\n"
            f"Accept: */*\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        )
        sock.sendall(request.encode())

        # Receive response
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

        sock.close()

        # Parse response
        response_str = response.decode("utf-8", errors="replace")
        lines = response_str.split("\r\n")

        # Parse status line
        if lines:
            status_parts = lines[0].split(" ", 2)
            if len(status_parts) >= 2:
                result["status_code"] = int(status_parts[1])

        # Parse headers
        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if line == "":
                body_start = i + 1
                break
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip()] = value.strip()

        result["headers"] = headers
        result["body"] = "\r\n".join(lines[body_start:])[:1000]  # First 1000 chars
        result["success"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    """Main diagnostic function."""
    host = "chatgpt.com"

    print_section("DNS Diagnostic for " + host)

    # 1. Check CUSTOM_DNS_HOSTS
    print_section("1. CUSTOM_DNS_HOSTS Configuration")
    custom_ip = get_custom_dns_ip()

    # 2. Check system DNS
    print_section("2. System DNS Resolution")
    system_ip = get_system_dns_ip()

    # 3. Check Google DNS
    print_section("3. Google DNS Resolution (8.8.8.8)")
    google_ip = get_google_dns_ip()

    # 4. Test connections
    print_section("4. Connection Tests")

    ips_to_test = []
    if custom_ip:
        ips_to_test.append(("CUSTOM_DNS_HOSTS", custom_ip))
    if system_ip and system_ip != custom_ip:
        ips_to_test.append(("System DNS", system_ip))
    if google_ip and google_ip != custom_ip and google_ip != system_ip:
        ips_to_test.append(("Google DNS", google_ip))

    for name, ip in ips_to_test:
        print(f"\nTesting {name}: {ip}")
        result = test_connection(ip, host)

        if result["success"]:
            print(f"  ✓ Connection successful")
            print(f"  Status: {result['status_code']}")
            print(f"  Headers: {len(result['headers'])} headers")
            if "server" in result["headers"]:
                print(f"  Server: {result['headers']['server']}")
            if "x-powered-by" in result["headers"]:
                print(f"  Powered by: {result['headers']['x-powered-by']}")
            print(f"  Body preview: {result['body'][:200]}...")
        else:
            print(f"  ✗ Connection failed: {result['error']}")

    # 5. Summary
    print_section("5. Summary")
    print(f"Host: {host}")
    print(f"CUSTOM_DNS_HOSTS IP: {custom_ip}")
    print(f"System DNS IP: {system_ip}")
    print(f"Google DNS IP: {google_ip}")

    if custom_ip == "151.101.1.195":
        print("\n⚠️  WARNING: CUSTOM_DNS_HOSTS uses Fastly CDN IP (151.101.1.195)")
        print("   This IP is likely incorrect for Azure App Service.")
        print("   Azure App Services use dynamic IPs that change.")
        print("\n   RECOMMENDATION:")
        print("   1. Remove the hardcoded IP from CUSTOM_DNS_HOSTS")
        print("   2. Set value to None to use DNS resolver instead")
        print("   3. Or use the IP from Google DNS if it works")


if __name__ == "__main__":
    main()
