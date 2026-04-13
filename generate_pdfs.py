"""Temporary script to generate sample compliance PDFs. Delete after use."""
from fpdf import FPDF
import os

docs_dir = "documents"
os.makedirs(docs_dir, exist_ok=True)

pdfs = {
    "compliance-policy-2024.pdf": [
        ("title", "Corporate Compliance Policy 2024"),
        ("", ""),
        ("h", "1. Purpose"),
        ("p", "This policy establishes the compliance framework for all business units within the organization. It ensures adherence to applicable laws, regulations, and internal standards governing financial services operations."),
        ("", ""),
        ("h", "2. Scope"),
        ("p", "This policy applies to all employees, contractors, and third-party vendors across all departments including Trading, Operations, Risk Management, and Client Services."),
        ("", ""),
        ("h", "3. Regulatory Framework"),
        ("p", "The organization operates under the supervision of the following regulatory bodies:"),
        ("p", "  - Securities and Exchange Commission (SEC)"),
        ("p", "  - Financial Industry Regulatory Authority (FINRA)"),
        ("p", "  - Office of the Comptroller of the Currency (OCC)"),
        ("p", "  - Consumer Financial Protection Bureau (CFPB)"),
        ("", ""),
        ("h", "4. Code of Conduct"),
        ("p", "4.1 All employees must complete annual compliance training within 30 days of the training release date."),
        ("p", "4.2 Any potential conflict of interest must be disclosed to the Compliance Department within 5 business days."),
        ("p", "4.3 Personal trading accounts must be pre-cleared through the compliance portal before executing any transaction."),
        ("p", "4.4 Gifts and entertainment exceeding USD 100 must be reported and approved by the compliance officer."),
        ("", ""),
        ("h", "5. Anti-Money Laundering (AML)"),
        ("p", "5.1 All customer accounts must undergo Know Your Customer (KYC) verification before activation."),
        ("p", "5.2 Suspicious Activity Reports (SARs) must be filed within 30 calendar days of detection."),
        ("p", "5.3 Currency Transaction Reports (CTRs) are required for cash transactions exceeding USD 10,000."),
        ("p", "5.4 Enhanced Due Diligence (EDD) is mandatory for high-risk customers and Politically Exposed Persons (PEPs)."),
        ("", ""),
        ("h", "6. Data Privacy and Protection"),
        ("p", "6.1 Customer data must be classified as Confidential, Internal, or Public."),
        ("p", "6.2 Access to customer PII is restricted to authorized personnel with a documented business need."),
        ("p", "6.3 Data breaches must be reported to the Chief Compliance Officer within 24 hours of discovery."),
        ("p", "6.4 All data transfers to third parties require a signed Data Processing Agreement (DPA)."),
        ("", ""),
        ("h", "7. Whistleblower Protection"),
        ("p", "Employees who report compliance violations in good faith are protected from retaliation under this policy and applicable law."),
        ("", ""),
        ("h", "8. Penalties for Non-Compliance"),
        ("p", "Violations may result in disciplinary action up to and including termination, regulatory fines, and referral to law enforcement."),
        ("", ""),
        ("b", "Effective Date: January 1, 2024"),
        ("b", "Next Review: December 31, 2024"),
        ("b", "Approved by: Chief Compliance Officer"),
    ],
    "risk-management-framework-v2.pdf": [
        ("title", "Risk Management Framework v2.0 - 2025"),
        ("", ""),
        ("h", "1. Executive Summary"),
        ("p", "This framework defines the risk management strategy, governance structure, and methodologies for identifying, assessing, mitigating, and monitoring risks across the enterprise."),
        ("", ""),
        ("h", "2. Risk Categories"),
        ("", ""),
        ("h2", "2.1 Market Risk"),
        ("p", "Market risk arises from fluctuations in interest rates, foreign exchange rates, equity prices, and commodity prices. Value-at-Risk (VaR) is calculated daily with a 99% confidence interval over a 10-day holding period. The maximum acceptable VaR limit is USD 50 million."),
        ("", ""),
        ("h2", "2.2 Credit Risk"),
        ("p", "Credit risk is the potential for loss due to a borrower or counterparty failing to meet obligations. All credit exposures are reviewed quarterly. Single-name concentration limits are set at 5% of Tier 1 capital."),
        ("", ""),
        ("h2", "2.3 Operational Risk"),
        ("p", "Operational risk includes losses from inadequate processes, systems failures, human error, or external events. Key Risk Indicators (KRIs) are tracked monthly. Each business unit maintains a Risk and Control Self-Assessment (RCSA) updated semi-annually."),
        ("", ""),
        ("h2", "2.4 Liquidity Risk"),
        ("p", "The organization maintains a Liquidity Coverage Ratio (LCR) above 110% and a Net Stable Funding Ratio (NSFR) above 100%. Stress testing is performed quarterly under three scenarios: baseline, adverse, and severely adverse."),
        ("", ""),
        ("h", "3. Risk Governance"),
        ("p", "3.1 The Board Risk Committee meets quarterly to review the enterprise risk profile."),
        ("p", "3.2 The Chief Risk Officer (CRO) reports directly to the CEO and has dotted-line reporting to the Board."),
        ("p", "3.3 Each business unit has a designated Risk Manager responsible for first-line risk oversight."),
        ("", ""),
        ("h", "4. Risk Appetite Statement"),
        ("p", "The organization accepts moderate risk in pursuit of strategic objectives. Risk appetite is expressed through quantitative limits (VaR, credit exposure) and qualitative statements (zero tolerance for regulatory violations)."),
        ("", ""),
        ("h", "5. Stress Testing"),
        ("p", "Enterprise-wide stress tests are conducted quarterly. Results are reported to the Board Risk Committee and submitted to regulators as required under CCAR/DFAST requirements."),
        ("", ""),
        ("h", "6. Incident Escalation"),
        ("p", "Risk events exceeding USD 1 million must be escalated to the CRO within 4 hours. Events exceeding USD 10 million require immediate Board notification."),
        ("", ""),
        ("b", "Version: 2.0"),
        ("b", "Effective Date: March 1, 2025"),
        ("b", "Owner: Chief Risk Officer"),
    ],
    "q3-2024-audit-report.pdf": [
        ("title", "Q3 2024 Internal Audit Report"),
        ("", ""),
        ("h", "1. Audit Summary"),
        ("p", "The Internal Audit department completed 12 audits during Q3 2024 covering Trading Operations, Client Onboarding, IT Security, and Regulatory Reporting."),
        ("", ""),
        ("h", "2. Key Findings"),
        ("", ""),
        ("b", "Finding #1: KYC Documentation Gaps (HIGH)"),
        ("p", "During the review of 500 client accounts, 37 accounts (7.4%) were found to have incomplete or expired KYC documentation. Of these, 12 accounts belonged to high-risk customers requiring Enhanced Due Diligence."),
        ("p", "Recommendation: Implement automated KYC expiry alerts and block trading for accounts with expired documentation."),
        ("p", "Remediation Deadline: November 30, 2024"),
        ("", ""),
        ("b", "Finding #2: Trade Surveillance Gaps (MEDIUM)"),
        ("p", "The trade surveillance system failed to flag 3 instances of potential wash trading in the equities desk during July 2024. Root cause was an outdated detection rule that did not account for new trading patterns."),
        ("p", "Recommendation: Update surveillance rules and perform a lookback review of Q2-Q3 equity trades."),
        ("p", "Remediation Deadline: October 31, 2024"),
        ("", ""),
        ("b", "Finding #3: Access Control Weaknesses (MEDIUM)"),
        ("p", "15 terminated employees retained active system access for an average of 8 days post-termination. Two accounts had access to sensitive financial data."),
        ("p", "Recommendation: Automate access revocation through HR-IT integration. Target SLA: access removed within 24 hours of termination."),
        ("p", "Remediation Deadline: December 15, 2024"),
        ("", ""),
        ("b", "Finding #4: Regulatory Report Timeliness (LOW)"),
        ("p", "2 out of 45 regulatory reports were submitted within 24 hours of the deadline. While no deadlines were missed, the margin is insufficient."),
        ("p", "Recommendation: Establish a T-5 business day internal deadline for all regulatory submissions."),
        ("p", "Remediation Deadline: October 15, 2024"),
        ("", ""),
        ("h", "3. Audit Statistics"),
        ("p", "Total Audits Completed: 12"),
        ("p", "High Severity Findings: 1"),
        ("p", "Medium Severity Findings: 4"),
        ("p", "Low Severity Findings: 7"),
        ("p", "Outstanding Remediation Items from Prior Quarters: 3"),
        ("", ""),
        ("h", "4. Management Response"),
        ("p", "Management has accepted all findings and committed to the proposed remediation timelines. Progress will be tracked in the monthly Compliance Dashboard."),
        ("", ""),
        ("b", "Report Date: October 10, 2024"),
        ("b", "Prepared by: Internal Audit Department"),
    ],
    "soe-45678-incident-report.pdf": [
        ("title", "Compliance Incident Report - SOE-45678"),
        ("", ""),
        ("h", "1. Incident Overview"),
        ("b", "Incident ID: SOE-45678"),
        ("b", "Date of Detection: August 15, 2024"),
        ("b", "Date of Occurrence: August 12-14, 2024"),
        ("b", "Reported By: Trade Surveillance Team"),
        ("b", "Severity: HIGH"),
        ("", ""),
        ("h", "2. Description"),
        ("p", "A series of unauthorized cross-trades were detected between two client accounts managed by Senior Trader John Smith (Employee ID: EMP-2847). The trades were executed outside normal market hours through the direct market access (DMA) system, bypassing standard pre-trade compliance checks."),
        ("", ""),
        ("p", "Total Value of Unauthorized Trades: USD 2,340,000"),
        ("p", "Number of Transactions: 7"),
        ("p", "Affected Accounts: ACC-78901, ACC-78902"),
        ("", ""),
        ("h", "3. Root Cause Analysis"),
        ("p", "3.1 The DMA system had a configuration gap that allowed after-hours order submission without triggering compliance alerts."),
        ("p", "3.2 The trader override permissions had not been reviewed since onboarding (18 months prior)."),
        ("p", "3.3 The daily trade reconciliation process runs at T+1, creating a window for undetected activity."),
        ("", ""),
        ("h", "4. Regulatory Impact"),
        ("p", "This incident constitutes a potential violation of:"),
        ("p", "  - SEC Rule 10b-5 (Anti-fraud provisions)"),
        ("p", "  - FINRA Rule 3110 (Supervision)"),
        ("p", "  - Internal Trading Policy Section 4.3"),
        ("", ""),
        ("p", "A preliminary SAR has been filed. Full regulatory disclosure is pending legal review."),
        ("", ""),
        ("h", "5. Immediate Actions Taken"),
        ("p", "5.1 Trader placed on administrative leave pending investigation."),
        ("p", "5.2 DMA after-hours access disabled for all traders."),
        ("p", "5.3 All trades reversed and client accounts made whole."),
        ("p", "5.4 Compliance alert rules updated to flag after-hours DMA activity."),
        ("", ""),
        ("h", "6. Remediation Plan"),
        ("p", "6.1 Implement real-time trade monitoring for DMA channels (Target: September 30, 2024)"),
        ("p", "6.2 Quarterly review of trader override permissions (Starting Q4 2024)"),
        ("p", "6.3 Reduce reconciliation window from T+1 to T+0 (Target: November 2024)"),
        ("p", "6.4 Mandatory re-training on trading policies for all desk personnel"),
        ("", ""),
        ("h", "7. Financial Impact"),
        ("p", "Direct Loss: USD 0 (trades reversed)"),
        ("p", "Estimated Remediation Cost: USD 150,000"),
        ("p", "Potential Regulatory Fine: USD 500,000 - USD 2,000,000 (pending)"),
        ("", ""),
        ("b", "Status: Under Investigation"),
        ("b", "Case Owner: Chief Compliance Officer"),
        ("b", "Last Updated: September 5, 2024"),
    ],
}


def write_line(pdf, style, text):
    if style == "title":
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT")
    elif style == "h":
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
    elif style == "h2":
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
    elif style == "b":
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
    elif style == "p":
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, text)
        pdf.set_x(pdf.l_margin)
    else:
        pdf.ln(4)


for filename, lines in pdfs.items():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)

    for style, text in lines:
        write_line(pdf, style, text)

    filepath = os.path.join(docs_dir, filename)
    pdf.output(filepath)
    print(f"Created: {filepath}")

print("Done! You can delete this script now.")
