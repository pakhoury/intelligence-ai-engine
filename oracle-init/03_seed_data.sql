-- Seed sample compliance data
ALTER SESSION SET CURRENT_SCHEMA = compliance_user;

-- ===================== COMPLIANCE_VIOLATIONS =====================
INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-45678', 'John Smith', 'Trading', DATE '2024-08-12', 'Unauthorized Trading', 'Critical', 'Escalated', NULL, 'SEC', 2340000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-11234', 'Alice Johnson', 'Operations', DATE '2024-06-15', 'KYC', 'High', 'Closed', DATE '2024-07-20', 'FINRA', 50000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-22345', 'Bob Williams', 'Client Services', DATE '2024-07-03', 'Data Privacy', 'Medium', 'In Progress', NULL, 'CFPB', 0);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-33456', 'Carol Davis', 'Risk Management', DATE '2024-05-20', 'Conflict of Interest', 'Low', 'Closed', DATE '2024-06-10', NULL, 0);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-44567', 'David Chen', 'Trading', DATE '2024-09-01', 'AML', 'High', 'Open', NULL, 'SEC', 175000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-55678', 'Emma Wilson', 'Operations', DATE '2024-03-18', 'KYC', 'Medium', 'Closed', DATE '2024-04-25', 'FINRA', 25000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-66789', 'Frank Miller', 'Compliance', DATE '2024-10-05', 'Unauthorized Trading', 'High', 'In Progress', NULL, 'SEC', 890000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-77890', 'Grace Lee', 'Trading', DATE '2024-01-22', 'AML', 'Critical', 'Closed', DATE '2024-03-15', 'SEC', 1500000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-88901', 'Henry Brown', 'Client Services', DATE '2024-04-11', 'Data Privacy', 'Low', 'Closed', DATE '2024-05-01', 'CFPB', 10000);

INSERT INTO COMPLIANCE_VIOLATIONS (SOE_ID, EMPLOYEE_NAME, DEPARTMENT, VIOLATION_DATE, VIOLATION_TYPE, SEVERITY, STATUS, CLOSURE_DATE, REGULATORY_BODY, FINANCIAL_IMPACT)
VALUES ('SOE-99012', 'Irene Taylor', 'Operations', DATE '2024-11-01', 'Conflict of Interest', 'Medium', 'Open', NULL, NULL, 0);

-- ===================== AUDIT_FINDINGS =====================
INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('Internal Audit', 'Client Services', DATE '2024-09-15', 'KYC documentation incomplete for 37 out of 500 reviewed accounts (7.4%). 12 accounts belonged to high-risk customers requiring Enhanced Due Diligence.', 'High', 'Open');

INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('Internal Audit', 'Trading', DATE '2024-09-15', 'Trade surveillance system failed to flag 3 instances of potential wash trading in equities desk during July 2024.', 'Medium', 'In Progress');

INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('Internal Audit', 'IT Security', DATE '2024-09-15', '15 terminated employees retained active system access for average of 8 days post-termination. Two had access to sensitive financial data.', 'Medium', 'In Progress');

INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('Internal Audit', 'Compliance', DATE '2024-09-15', '2 out of 45 regulatory reports submitted within 24 hours of deadline. No deadlines missed but margin is insufficient.', 'Low', 'Closed');

INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('External Audit', 'Operations', DATE '2024-06-20', 'Vendor risk assessments not updated for 8 out of 25 critical vendors in the past 12 months.', 'Medium', 'Open');

INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('Regulatory Inspection', 'Trading', DATE '2024-04-10', 'Best execution documentation insufficient for OTC derivatives trades exceeding USD 1M.', 'High', 'In Progress');

INSERT INTO AUDIT_FINDINGS (AUDIT_TYPE, DEPARTMENT, FINDING_DATE, DESCRIPTION, RISK_RATING, STATUS)
VALUES ('Internal Audit', 'Risk Management', DATE '2024-07-22', 'Stress testing scenarios not updated to reflect current market conditions. Last update was Q1 2024.', 'Medium', 'Closed');

-- ===================== CONTROL_MAPPINGS =====================
INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('KYC Verification', 'Client Services', 'AML/BSA - Customer Identification Program', DATE '2023-01-01', 'Daily');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('Trade Surveillance Monitoring', 'Compliance', 'FINRA Rule 3110 - Supervision', DATE '2023-01-01', 'Daily');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('SAR Filing Process', 'Compliance', 'BSA - Suspicious Activity Reporting', DATE '2023-06-15', 'Monthly');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('Access Revocation on Termination', 'IT Security', 'SOX Section 404 - IT General Controls', DATE '2024-01-01', 'Weekly');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('Data Privacy Impact Assessment', 'Legal', 'CFPB Regulation P - Privacy of Consumer Financial Information', DATE '2023-03-01', 'Quarterly');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('VaR Limit Monitoring', 'Risk Management', 'Basel III - Market Risk Framework', DATE '2023-01-01', 'Daily');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('Regulatory Report Submission', 'Compliance', 'SEC/FINRA - Periodic Reporting Requirements', DATE '2023-01-01', 'Monthly');

INSERT INTO CONTROL_MAPPINGS (CONTROL_NAME, CONTROL_OWNER_DEPT, COMPLIANCE_REQUIREMENT, EFFECTIVE_DATE, TEST_FREQUENCY)
VALUES ('Vendor Risk Assessment', 'Operations', 'OCC Guidance on Third-Party Risk Management', DATE '2024-06-01', 'Quarterly');

-- ===================== RISK_EVENTS =====================
INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-45678', 'Trading', DATE '2024-08-12', 'Operational', 150000, 'Unauthorized cross-trades via DMA system bypassing pre-trade compliance checks. 7 transactions totaling USD 2.34M.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-77890', 'Trading', DATE '2024-01-22', 'Regulatory', 1500000, 'AML violation - failure to file SAR for suspicious wire transfers within required timeframe.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-11234', 'Operations', DATE '2024-06-15', 'Operational', 50000, 'Expired KYC documentation for high-risk customer accounts led to regulatory penalty.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-33456', 'Risk Management', DATE '2024-05-20', 'Market', 320000, 'VaR limit breach on fixed income desk due to unhedged interest rate exposure.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-55678', 'IT Security', DATE '2024-02-08', 'Operational', 75000, 'System outage during market hours caused by failed database migration. Trading halted for 45 minutes.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-44567', 'Trading', DATE '2024-09-01', 'Regulatory', 175000, 'Failure to report large currency transactions exceeding USD 10,000 threshold.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-66789', 'Compliance', DATE '2024-10-05', 'Operational', 890000, 'Rogue trading detected - unauthorized proprietary positions taken outside approved risk limits.');

INSERT INTO RISK_EVENTS (SOE_ID, DEPARTMENT, EVENT_DATE, RISK_CATEGORY, LOSS_AMOUNT, EVENT_DESCRIPTION)
VALUES ('SOE-22345', 'Client Services', DATE '2024-07-03', 'Operational', 0, 'Customer PII exposed via misconfigured API endpoint. No financial loss but regulatory reporting required.');

COMMIT;
