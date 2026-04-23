from __future__ import annotations

import argparse
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def detect_primary_ipv4() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a self-signed local HTTPS certificate for LAN testing.")
    parser.add_argument("--ip", type=str, default=None, help="Primary LAN IPv4 to include in the certificate SAN")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/https"), help="Where to write cert and key")
    parser.add_argument("--days", type=int, default=30, help="Certificate validity in days")
    args = parser.parse_args()

    lan_ip = args.ip or detect_primary_ipv4()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "QNA-Auth Local"),
            x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
        ]
    )
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=max(1, args.days)))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.IPAddress(ipaddress.ip_address(lan_ip)),
                    x509.DNSName("localhost"),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_path = args.output_dir / "qna_auth_local.crt"
    key_path = args.output_dir / "qna_auth_local.key"

    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    print(f"LAN IP: {lan_ip}")
    print(f"Certificate: {cert_path.resolve()}")
    print(f"Private key: {key_path.resolve()}")
    print("Use these env vars before starting the server:")
    print(f'$env:QNA_AUTH_SSL_CERTFILE="{cert_path.resolve()}"')
    print(f'$env:QNA_AUTH_SSL_KEYFILE="{key_path.resolve()}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
