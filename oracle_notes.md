# Oracle Database Notes

- Use Oracle Transparent Data Encryption (TDE) to encrypt data at rest.
- Use TLS/SSL for connections to Oracle (configure wallet or set SSL settings).
- Use least-privilege roles for database accounts. Do not connect as SYS.
- Example Python connector: cx_Oracle (ororacledb)
- Ensure network access is restricted by firewall and only application servers can connect.
