-- Create application user/schema
ALTER SESSION SET "_ORACLE_SCRIPT"=true;

CREATE USER compliance_user IDENTIFIED BY compliance_pass
    DEFAULT TABLESPACE users
    TEMPORARY TABLESPACE temp
    QUOTA UNLIMITED ON users;

GRANT CONNECT, RESOURCE TO compliance_user;
GRANT CREATE SESSION TO compliance_user;
GRANT CREATE TABLE TO compliance_user;
GRANT CREATE VIEW TO compliance_user;
GRANT CREATE SEQUENCE TO compliance_user;

ALTER SESSION SET "_ORACLE_SCRIPT"=false;
