from kafka import KafkaConsumer
import json
import pandas as pd
import csv
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
import threading
import os
import numpy as np

class BankingAnalyticsConsumer:
    def __init__(self):
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            'banking_transactions',
            bootstrap_servers='localhost:9092',
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='banking-analysis-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        # Initialize CSV file with comprehensive headers including sender bank name
        self.csv_filename = 'backend/bank.csv'
        self.csv_headers = [
            # Bank Statement Core Fields
            'Transaction_Sequence_Number', 'Statement_Period_Start', 'Statement_Period_End', 'Statement_Number',
            
            # Core Transaction Details
            'Transaction_Ref_No', 'Account_Number', 'Account_Type', 'Account_Type_Name', 
            'Account_Interest_Rate', 'Account_Min_Balance', 'Account_Transaction_Limit',
            'Transaction_Date', 'Transaction_Time', 'Transaction_DateTime', 'Value_Date', 'Processing_Date',
            
            # Time-based Analytics
            'Month_Category', 'Week_In_Month', 'Day_Of_Week', 'Quarter', 'Hour_Of_Day',
            'Is_Weekend', 'Is_Holiday',
            
            # Transaction Classification
            'Transaction_Code', 'Transaction_Description', 'Transaction_Category', 
            'Transaction_Sub_Type', 'Debit_Credit_Flag', 'Transaction_Purpose',
            'Realistic_Transaction_Description',
            
            # Financial Details (Bank Statement Format)
            'Transaction_Amount', 'Transaction_Currency', 'Exchange_Rate', 'USD_Equivalent',
            'Opening_Balance', 'Closing_Balance', 'Available_Balance',
            
            # Fee and Charges
            'Charges_Applied', 'GST_on_Charges', 'Total_Charges', 'Net_Amount',
            
            # Beneficiary Information
            'Beneficiary_Account', 'Beneficiary_IFSC', 'Beneficiary_Name', 'Beneficiary_Bank',
            
            # Banking Identifiers (Enhanced for Bank Statements)
            'UTR_Number', 'Cheque_Number', 'Card_Number', 'UPI_ID',
            
            # Channel Information
            'Channel_Type', 'Channel_Name', 'Device_Type', 'IP_Address', 'User_Agent',
            
            # Branch and Location
            'Branch_Code', 'Branch_Name', 'Transaction_State', 'Transaction_City',
            'Region', 'Zone', 'Country',
            
            # Customer Information
            'Customer_ID', 'Customer_Name', 'Customer_Type', 'Customer_Segment',
            'KYC_Status', 'Customer_Risk_Category', 'Relationship_Manager',
            
            # Bank Information (Enhanced - Separate sender bank columns)
            'Sender_Bank_Name', 'Sender_Bank_Code', 'Sender_IFSC', 'Receiver_Bank_Name', 'Receiver_Bank_Code',
            
            # Transaction Status and Processing
            'Transaction_Status', 'Status_Code', 'Status_Description', 'Processing_Time_Seconds',
            'Retry_Count',
            
            # Compliance and Regulatory
            'AML_Flag', 'CTR_Required', 'STR_Flag', 'Regulatory_Reporting_Required',
            
            # Risk Assessment
            'Risk_Score', 'Risk_Level', 'Risk_Flags',
            
            # Additional Identifiers
            'Batch_ID', 'Journal_ID', 'Sequence_Number', 'Business_Date',
            
            # Technical Metadata
            'Record_Created_At', 'Record_Updated_At', 'Data_Source', 'Message_ID',
            'Correlation_ID', 'Session_ID'
        ]
        
        # Initialize tracking variables
        self.transaction_count = 0
        self.anomaly_stats = defaultdict(int)
        self.records_with_anomalies = 0
        
        # CRITICAL: Unique tracking for protected fields (NO DUPLICATES ALLOWED)
        self.unique_transaction_refs = set()
        self.unique_account_numbers = set()
        self.unique_customer_ids = set()
        self.unique_statement_numbers = set()
        self.duplicate_transaction_refs = 0
        self.duplicate_account_numbers = 0
        self.duplicate_customer_ids = 0
        self.duplicate_statement_numbers = 0
        
        # Bank statement specific tracking
        self.customer_statement_tracking = defaultdict(dict)
        self.customer_sequence_tracking = defaultdict(list)
        self.customer_balance_progression = defaultdict(list)
        self.utr_numbers = set()
        self.cheque_numbers = set()
        
        # Customer frequency tracking (enhanced for bank statements)
        self.customer_transaction_count = defaultdict(int)
        self.customer_names = {}
        self.customer_account_mapping = {}
        self.customer_balances = {}
        self.customer_daily_activity = defaultdict(set)
        self.customer_monthly_summary = defaultdict(lambda: {'transactions': 0, 'total_amount': 0, 'avg_balance': 0})
        
        # Enhanced sender bank tracking
        self.sender_bank_stats = defaultdict(int)
        self.bank_to_bank_transfers = defaultdict(int)
        self.sender_bank_transaction_volume = defaultdict(float)
        
        # Real-time analytics tracking (enhanced with bank statement metrics)
        self.real_time_stats = {
            'total_transactions': 0,
            'current_month_transactions': 0,
            'last_month_transactions': 0,
            'total_debit_amount': 0.0,
            'total_credit_amount': 0.0,
            'channel_stats': defaultdict(int),
            'category_stats': defaultdict(int),
            'risk_stats': defaultdict(int),
            'customer_segment_stats': defaultdict(int),
            'geography_stats': defaultdict(int),
            'hourly_stats': defaultdict(int),
            'daily_stats': defaultdict(int),
            'weekly_stats': defaultdict(int),
            'currency_stats': defaultdict(float),
            'bank_stats': defaultdict(int),
            'sender_bank_stats': defaultdict(int),  # Enhanced sender bank tracking
            'receiver_bank_stats': defaultdict(int),  # NEW: Receiver bank tracking
            'status_stats': defaultdict(int),
            'account_type_stats': defaultdict(int),
            'weekend_transactions': 0,
            'holiday_transactions': 0,
            'high_value_transactions': 0,
            'regulatory_transactions': 0,
            'failed_transactions': 0,
            'aml_flagged_transactions': 0,
            'suspicious_transactions': 0,
            'unique_customers_active': 0,
            'customer_frequency_buckets': defaultdict(int),
            'unique_utr_numbers': 0,
            'unique_cheque_numbers': 0,
            'transactions_with_beneficiary': 0,
            'value_date_differences': 0,
            'statement_periods_active': 0,
            'sequential_numbering_issues': 0,
            'balance_inconsistencies': 0,
            'inter_bank_transfers': 0,  # NEW: Track inter-bank transfers
            'intra_bank_transfers': 0,  # NEW: Track intra-bank transfers
            # CRITICAL: Protected field integrity stats
            'protected_field_integrity': {
                'transaction_ref_duplicates': 0,
                'account_number_duplicates': 0,
                'customer_id_duplicates': 0,
                'statement_number_duplicates': 0,
                'null_transaction_refs': 0,
                'null_account_numbers': 0,
                'null_customer_ids': 0,
                'null_statement_numbers': 0,
                'null_sender_bank_names': 0,  # NEW: Track null sender bank names
                'invalid_sender_bank_names': 0  # NEW: Track invalid sender bank names
            }
        }
        
        # Recent transactions for real-time monitoring (last 100)
        self.recent_transactions = deque(maxlen=100)
        
        # Enhanced bank-to-bank transfer tracking
        self.bank_transfer_matrix = defaultdict(lambda: defaultdict(int))
        self.bank_transfer_volume_matrix = defaultdict(lambda: defaultdict(float))
        
        # Start real-time dashboard thread
        self.dashboard_active = True
        self.dashboard_thread = threading.Thread(target=self.real_time_dashboard)
        self.dashboard_thread.daemon = True

    def safe_format(self, value, format_spec=""):
        """Safely format values, handling None values"""
        if value is None:
            return "None"
        try:
            if format_spec:
                return f"{value:{format_spec}}"
            else:
                return str(value)
        except:
            return str(value)

    def clean_anomaly_characters(self, value):
        """Clean anomaly characters from string values"""
        if isinstance(value, str) and '@#$' in value:
            return value.replace('@#$%^&*()', '').strip()
        return value

    def validate_protected_fields(self, transaction_data):
        """CRITICAL: Validate protected fields for uniqueness and non-null values"""
        validation_errors = []
        
        # PROTECTED FIELDS - MUST BE UNIQUE AND NON-NULL
        protected_fields = {
            'Transaction_Ref_No': self.unique_transaction_refs,
            'Account_Number': self.unique_account_numbers,
            'Customer_ID': self.unique_customer_ids,
            'Statement_Number': self.unique_statement_numbers
        }
        
        for field_name, unique_set in protected_fields.items():
            field_value = transaction_data.get(field_name)
            
            # Check for null/empty values
            if field_value is None or field_value == "" or str(field_value).strip() == "":
                validation_errors.append(f"NULL_{field_name}")
                self.real_time_stats['protected_field_integrity'][f'null_{field_name.lower()}s'] += 1
                continue
            
            # Check for duplicates
            if field_value in unique_set:
                validation_errors.append(f"DUPLICATE_{field_name}")
                self.real_time_stats['protected_field_integrity'][f'{field_name.lower()}_duplicates'] += 1
                
                # Increment specific duplicate counters
                if field_name == 'Transaction_Ref_No':
                    self.duplicate_transaction_refs += 1
                elif field_name == 'Account_Number':
                    self.duplicate_account_numbers += 1
                elif field_name == 'Customer_ID':
                    self.duplicate_customer_ids += 1
                elif field_name == 'Statement_Number':
                    self.duplicate_statement_numbers += 1
            else:
                # Add to unique set if valid
                unique_set.add(field_value)
        
        # NEW: Validate sender bank name
        sender_bank_name = transaction_data.get('Sender_Bank_Name')
        if sender_bank_name is None or sender_bank_name == "":
            validation_errors.append("NULL_SENDER_BANK_NAME")
            self.real_time_stats['protected_field_integrity']['null_sender_bank_names'] += 1
        elif not isinstance(sender_bank_name, str) or len(sender_bank_name.strip()) < 3:
            validation_errors.append("INVALID_SENDER_BANK_NAME")
            self.real_time_stats['protected_field_integrity']['invalid_sender_bank_names'] += 1
        
        return validation_errors

    def detect_comprehensive_anomalies(self, transaction_data):
        """Enhanced anomaly detection for comprehensive banking data including bank statement format"""
        anomalies = []
        
        # FIRST: Validate protected fields (CRITICAL)
        protected_field_errors = self.validate_protected_fields(transaction_data)
        anomalies.extend(protected_field_errors)
        
        # PROTECTED FIELDS - NEVER check for anomalies in these critical fields (they are validated above)
        protected_fields = [
            'Transaction_Ref_No', 'Customer_ID', 'Account_Number', 'Customer_Name',
            'UTR_Number', 'Cheque_Number', 'Transaction_Sequence_Number', 'Statement_Number',
            'Sender_Bank_Name', 'Sender_Bank_Code', 'Sender_IFSC'  # NEW: Protect sender bank fields
        ]
        
        # Check for null/empty values in critical fields (except protected ones)
        critical_fields = [
            'Transaction_Date', 'Transaction_Amount', 'Account_Type',
            'Transaction_Status', 'Transaction_Code', 'Opening_Balance', 'Closing_Balance'
        ]
        
        for field in critical_fields:
            if field in transaction_data:
                value = transaction_data[field]
                if value is None:
                    anomalies.append(f"NULL_{field}")
                elif isinstance(value, str) and value == "":
                    anomalies.append(f"EMPTY_{field}")
                elif isinstance(value, str) and "@#$" in value:
                    anomalies.append(f"SPECIAL_CHARS_{field}")
        
        # Bank statement specific validations
        
        # Sequential numbering validation
        seq_num = transaction_data.get('Transaction_Sequence_Number')
        if seq_num is not None and isinstance(seq_num, (int, float)) and seq_num <= 0:
            anomalies.append("INVALID_SEQUENCE_NUMBER")
        
        # Balance consistency checks (bank statement requirement)
        opening_bal = transaction_data.get('Opening_Balance')
        closing_bal = transaction_data.get('Closing_Balance')
        txn_amount = transaction_data.get('Transaction_Amount')
        dr_cr_flag = transaction_data.get('Debit_Credit_Flag')
        
        if all(x is not None for x in [opening_bal, closing_bal, txn_amount, dr_cr_flag]):
            if isinstance(opening_bal, (int, float)) and isinstance(closing_bal, (int, float)) and isinstance(txn_amount, (int, float)):
                expected_closing = opening_bal + txn_amount if dr_cr_flag == 'CR' else opening_bal - txn_amount
                if abs(closing_bal - expected_closing) > 0.01:  # Allow for rounding
                    anomalies.append("BALANCE_INCONSISTENCY")
                    self.real_time_stats['balance_inconsistencies'] += 1
        
        # Value date validation
        txn_date = transaction_data.get('Transaction_Date')
        value_date = transaction_data.get('Value_Date')
        if txn_date and value_date and isinstance(txn_date, str) and isinstance(value_date, str):
            try:
                txn_dt = datetime.strptime(txn_date, '%Y-%m-%d')
                val_dt = datetime.strptime(value_date, '%Y-%m-%d')
                if (val_dt - txn_dt).days > 5:  # Value date more than 5 days after transaction date
                    anomalies.append("EXCESSIVE_VALUE_DATE_DIFFERENCE")
            except ValueError:
                pass  # Date format issues will be caught elsewhere
        
        # Account type validation
        valid_account_types = ['SAVINGS', 'CURRENT', 'FIXED_DEPOSIT', 'RECURRING_DEPOSIT', 'SALARY', 'NRI']
        account_type = transaction_data.get('Account_Type')
        if account_type and account_type not in valid_account_types:
            anomalies.append("INVALID_ACCOUNT_TYPE")
        
        # Financial anomalies
        amount = transaction_data.get('Transaction_Amount')
        if amount is not None:
            if isinstance(amount, (int, float)):
                if amount < 0:
                    anomalies.append("NEGATIVE_AMOUNT")
                elif amount > 10000000:  # > 1 crore
                    anomalies.append("EXTREME_HIGH_AMOUNT")
                elif amount == 0:
                    anomalies.append("ZERO_AMOUNT")
        
        # Account type specific validations
        account_min_balance = transaction_data.get('Account_Min_Balance')
        if (account_min_balance is not None and closing_bal is not None and 
            isinstance(account_min_balance, (int, float)) and isinstance(closing_bal, (int, float))):
            if closing_bal < account_min_balance:
                anomalies.append("BELOW_MIN_BALANCE")
        
        # Transaction limit validation
        account_txn_limit = transaction_data.get('Account_Transaction_Limit')
        if (account_txn_limit is not None and amount is not None and 
            isinstance(account_txn_limit, (int, float)) and isinstance(amount, (int, float))):
            if amount > account_txn_limit:
                anomalies.append("EXCEEDS_TRANSACTION_LIMIT")
        
        # Negative balance check
        if isinstance(closing_bal, (int, float)) and closing_bal < 0:
            anomalies.append("NEGATIVE_CLOSING_BALANCE")
        
        # Transaction code validation
        valid_codes = ['NEFT', 'RTGS', 'IMPS', 'UPI', 'NACH', 'CHQS', 'CASH', 'CARD', 'NETB', 'MOBP']
        txn_code = transaction_data.get('Transaction_Code')
        if txn_code and txn_code not in valid_codes:
            anomalies.append("INVALID_TRANSACTION_CODE")
        
        # DR/CR flag validation
        if dr_cr_flag and dr_cr_flag not in ['DR', 'CR']:
            anomalies.append("INVALID_DR_CR_FLAG")
        
        # Status validation
        valid_statuses = ['SUCCESS', 'PENDING', 'FAILED', 'RETURNED']
        status = transaction_data.get('Transaction_Status')
        if status and status not in valid_statuses:
            anomalies.append("INVALID_STATUS")
        
        # Date format validation
        date_fields = ['Transaction_Date', 'Value_Date', 'Processing_Date', 'Statement_Period_Start', 'Statement_Period_End']
        for field in date_fields:
            date_value = transaction_data.get(field)
            if date_value and isinstance(date_value, str):
                try:
                    datetime.strptime(date_value, '%Y-%m-%d')
                except ValueError:
                    anomalies.append(f"INVALID_DATE_FORMAT_{field}")
        
        # Risk score validation
        risk_score = transaction_data.get('Risk_Score')
        if risk_score is not None:
            if not isinstance(risk_score, (int, float)) or risk_score < 0 or risk_score > 100:
                anomalies.append("INVALID_RISK_SCORE")
        
        # Customer segment validation
        valid_segments = ['PREMIUM', 'GOLD', 'SILVER', 'BASIC']
        customer_segment = transaction_data.get('Customer_Segment')
        if customer_segment and customer_segment not in valid_segments:
            anomalies.append("INVALID_CUSTOMER_SEGMENT")
        
        # Channel validation
        valid_channel_types = ['DIGITAL', 'PHYSICAL', 'ASSISTED']
        channel_type = transaction_data.get('Channel_Type')
        if channel_type and channel_type not in valid_channel_types:
            anomalies.append("INVALID_CHANNEL_TYPE")
        
        # Currency validation
        currency = transaction_data.get('Transaction_Currency')
        if currency and currency not in ['INR', 'USD', 'EUR', 'GBP', 'AED']:
            anomalies.append("INVALID_CURRENCY")
        
        # Interest rate validation for account types
        interest_rate = transaction_data.get('Account_Interest_Rate')
        if interest_rate is not None and isinstance(interest_rate, (int, float)):
            if interest_rate < 0 or interest_rate > 15:  # Reasonable range for interest rates
                anomalies.append("INVALID_INTEREST_RATE")
        
        return anomalies

    def update_bank_statement_tracking(self, transaction_data):
        """Track bank statement specific metrics and patterns including sender bank analytics"""
        try:
            customer_id = transaction_data.get('Customer_ID')
            if not customer_id:
                return
            
            # Track statement information
            statement_number = transaction_data.get('Statement_Number')
            statement_start = transaction_data.get('Statement_Period_Start')
            statement_end = transaction_data.get('Statement_Period_End')
            
            if statement_number and customer_id not in self.customer_statement_tracking:
                self.customer_statement_tracking[customer_id] = {
                    'statement_number': statement_number,
                    'statement_start': statement_start,
                    'statement_end': statement_end,
                    'transaction_count': 0
                }
            
            if customer_id in self.customer_statement_tracking:
                self.customer_statement_tracking[customer_id]['transaction_count'] += 1
            
            # Track sequence numbers per customer
            seq_num = transaction_data.get('Transaction_Sequence_Number')
            if seq_num is not None:
                self.customer_sequence_tracking[customer_id].append(seq_num)
                
                # Check for sequential numbering issues
                if len(self.customer_sequence_tracking[customer_id]) > 1:
                    prev_seq = self.customer_sequence_tracking[customer_id][-2]
                    if seq_num != prev_seq + 1:
                        self.real_time_stats['sequential_numbering_issues'] += 1
            
            # Track balance progression
            opening_bal = transaction_data.get('Opening_Balance')
            closing_bal = transaction_data.get('Closing_Balance')
            if opening_bal is not None and closing_bal is not None:
                self.customer_balance_progression[customer_id].append({
                    'transaction_date': transaction_data.get('Transaction_Date'),
                    'opening_balance': opening_bal,
                    'closing_balance': closing_bal,
                    'transaction_amount': transaction_data.get('Transaction_Amount'),
                    'dr_cr_flag': transaction_data.get('Debit_Credit_Flag')
                })
            
            # Track UTR and Cheque numbers
            utr_number = transaction_data.get('UTR_Number')
            if utr_number:
                self.utr_numbers.add(utr_number)
                self.real_time_stats['unique_utr_numbers'] = len(self.utr_numbers)
            
            cheque_number = transaction_data.get('Cheque_Number')
            if cheque_number:
                self.cheque_numbers.add(cheque_number)
                self.real_time_stats['unique_cheque_numbers'] = len(self.cheque_numbers)
            
            # Track beneficiary information
            if transaction_data.get('Beneficiary_Account'):
                self.real_time_stats['transactions_with_beneficiary'] += 1
            
            # Track value date differences
            txn_date = transaction_data.get('Transaction_Date')
            value_date = transaction_data.get('Value_Date')
            if txn_date and value_date and txn_date != value_date:
                self.real_time_stats['value_date_differences'] += 1
            
            # Update statement periods count
            self.real_time_stats['statement_periods_active'] = len(self.customer_statement_tracking)
            
            # Enhanced sender bank tracking
            sender_bank_name = transaction_data.get('Sender_Bank_Name')
            receiver_bank_name = transaction_data.get('Receiver_Bank_Name')
            amount = transaction_data.get('Transaction_Amount', 0)
            
            if sender_bank_name:
                self.sender_bank_stats[sender_bank_name] += 1
                self.real_time_stats['sender_bank_stats'][sender_bank_name] += 1
                if isinstance(amount, (int, float)):
                    self.sender_bank_transaction_volume[sender_bank_name] += amount
            
            if receiver_bank_name:
                self.real_time_stats['receiver_bank_stats'][receiver_bank_name] += 1
            
            # Track bank-to-bank transfers with enhanced analytics
            if sender_bank_name and receiver_bank_name:
                transfer_key = f"{sender_bank_name} -> {receiver_bank_name}"
                self.bank_to_bank_transfers[transfer_key] += 1
                
                # Track in transfer matrix
                self.bank_transfer_matrix[sender_bank_name][receiver_bank_name] += 1
                if isinstance(amount, (int, float)):
                    self.bank_transfer_volume_matrix[sender_bank_name][receiver_bank_name] += amount
                
                # Classify as inter-bank or intra-bank transfer
                if sender_bank_name == receiver_bank_name:
                    self.real_time_stats['intra_bank_transfers'] += 1
                else:
                    self.real_time_stats['inter_bank_transfers'] += 1
            
        except Exception as e:
            pass  # Silently handle errors to avoid disrupting main processing

    def update_customer_tracking(self, transaction_data):
        """Track customer frequency and patterns for analytics (enhanced for bank statements)"""
        try:
            customer_id = transaction_data.get('Customer_ID')
            customer_name = transaction_data.get('Customer_Name')
            account_number = transaction_data.get('Account_Number')
            transaction_date = transaction_data.get('Transaction_Date')
            
            if customer_id:
                # Update customer transaction count
                self.customer_transaction_count[customer_id] += 1
                
                # Store customer name mapping
                if customer_name:
                    self.customer_names[customer_id] = customer_name
                
                # Store customer account mapping (enhanced for bank statements)
                if account_number:
                    self.customer_account_mapping[customer_id] = {
                        'account_number': account_number,
                        'account_type': transaction_data.get('Account_Type'),
                        'account_type_name': transaction_data.get('Account_Type_Name'),
                        'customer_segment': transaction_data.get('Customer_Segment'),
                        'customer_type': transaction_data.get('Customer_Type'),
                        'account_interest_rate': transaction_data.get('Account_Interest_Rate'),
                        'account_min_balance': transaction_data.get('Account_Min_Balance'),
                        'statement_number': transaction_data.get('Statement_Number'),
                        'sender_bank_name': transaction_data.get('Sender_Bank_Name')
                    }
                
                # Track daily activity
                if transaction_date:
                    self.customer_daily_activity[customer_id].add(transaction_date)
                
                # Track balance progression
                closing_balance = transaction_data.get('Closing_Balance')
                if isinstance(closing_balance, (int, float)):
                    self.customer_balances[customer_id] = closing_balance
                
                # Track monthly summary
                month_category = transaction_data.get('Month_Category', 'Unknown')
                amount = transaction_data.get('Transaction_Amount', 0)
                if isinstance(amount, (int, float)):
                    self.customer_monthly_summary[customer_id]['transactions'] += 1
                    self.customer_monthly_summary[customer_id]['total_amount'] += amount
                    if closing_balance:
                        self.customer_monthly_summary[customer_id]['avg_balance'] = closing_balance
                
                # Update frequency bucket stats
                txn_count = self.customer_transaction_count[customer_id]
                
                # Reset all frequency buckets and recalculate
                self.real_time_stats['customer_frequency_buckets'] = defaultdict(int)
                for count in self.customer_transaction_count.values():
                    if count <= 5:
                        self.real_time_stats['customer_frequency_buckets']['1-5'] += 1
                    elif count <= 10:
                        self.real_time_stats['customer_frequency_buckets']['6-10'] += 1
                    elif count <= 20:
                        self.real_time_stats['customer_frequency_buckets']['11-20'] += 1
                    elif count <= 30:
                        self.real_time_stats['customer_frequency_buckets']['21-30'] += 1
                    else:
                        self.real_time_stats['customer_frequency_buckets']['30+'] += 1
                
                # Update unique customers active count
                self.real_time_stats['unique_customers_active'] = len(self.customer_transaction_count)
        
        except Exception as e:
            pass  # Silently handle errors

    def update_real_time_stats(self, transaction_data):
        """Update real-time statistics for dashboard including bank statement metrics"""
        try:
            self.real_time_stats['total_transactions'] += 1
            
            # Update customer tracking
            self.update_customer_tracking(transaction_data)
            
            # Update bank statement specific tracking
            self.update_bank_statement_tracking(transaction_data)
            
            # Month category stats
            month_category = transaction_data.get('Month_Category', '')
            if month_category == 'Current Month':
                self.real_time_stats['current_month_transactions'] += 1
            elif month_category == 'Last Month':
                self.real_time_stats['last_month_transactions'] += 1
            
            # Amount statistics
            amount = transaction_data.get('Transaction_Amount', 0)
            if isinstance(amount, (int, float)) and amount > 0:
                dr_cr_flag = transaction_data.get('Debit_Credit_Flag', '')
                if dr_cr_flag == 'DR':
                    self.real_time_stats['total_debit_amount'] += amount
                else:
                    self.real_time_stats['total_credit_amount'] += amount
                
                # High value transaction tracking
                if amount >= 1000000:  # >= 10 lakh
                    self.real_time_stats['high_value_transactions'] += 1
            
            # Account type statistics (clean and validate)
            account_type = self.clean_anomaly_characters(transaction_data.get('Account_Type', 'UNKNOWN'))
            if account_type and account_type != 'UNKNOWN':
                self.real_time_stats['account_type_stats'][account_type] += 1
            
            # Channel statistics (clean and validate)
            channel_name = self.clean_anomaly_characters(transaction_data.get('Channel_Name', 'UNKNOWN'))
            if channel_name and channel_name != 'UNKNOWN':
                self.real_time_stats['channel_stats'][channel_name] += 1
            
            # Category statistics (clean and validate)
            category = self.clean_anomaly_characters(transaction_data.get('Transaction_Category', 'UNKNOWN'))
            if category and category != 'UNKNOWN':
                self.real_time_stats['category_stats'][category] += 1
            
            # Risk statistics (clean and validate)
            risk_level = self.clean_anomaly_characters(transaction_data.get('Risk_Level', 'LOW'))
            if risk_level in ['LOW', 'MEDIUM', 'HIGH']:
                self.real_time_stats['risk_stats'][risk_level] += 1
            else:
                # Default fallback for any invalid risk levels
                self.real_time_stats['risk_stats']['LOW'] += 1
                risk_level = 'LOW'
            
            # Customer segment statistics (clean and validate)
            segment = self.clean_anomaly_characters(transaction_data.get('Customer_Segment', 'UNKNOWN'))
            if segment and segment != 'UNKNOWN':
                self.real_time_stats['customer_segment_stats'][segment] += 1
            
            # Geography statistics (clean and validate)
            state = self.clean_anomaly_characters(transaction_data.get('Transaction_State', 'UNKNOWN'))
            if state and state != 'UNKNOWN':
                self.real_time_stats['geography_stats'][state] += 1
            
            # Time-based statistics
            hour = transaction_data.get('Hour_Of_Day', 0)
            if isinstance(hour, (int, float)):
                self.real_time_stats['hourly_stats'][int(hour)] += 1
            
            day_of_week = self.clean_anomaly_characters(transaction_data.get('Day_Of_Week', 'UNKNOWN'))
            if day_of_week and day_of_week != 'UNKNOWN':
                self.real_time_stats['daily_stats'][day_of_week] += 1
            
            week_in_month = transaction_data.get('Week_In_Month', 0)
            if isinstance(week_in_month, (int, float)):
                self.real_time_stats['weekly_stats'][int(week_in_month)] += 1
            
            # Currency statistics
            currency = transaction_data.get('Transaction_Currency', 'INR')
            if isinstance(amount, (int, float)):
                self.real_time_stats['currency_stats'][currency] += amount
            
            # Bank statistics (from account number prefix) - Account_Number is protected, so always valid
            account_number = transaction_data.get('Account_Number', '')
            if account_number and len(account_number) >= 4:
                bank_prefix = account_number[:4]
                self.real_time_stats['bank_stats'][bank_prefix] += 1
            
            # Status statistics (clean and validate)
            status = self.clean_anomaly_characters(transaction_data.get('Transaction_Status', 'UNKNOWN'))
            if status and status != 'UNKNOWN':
                self.real_time_stats['status_stats'][status] += 1
            
            # Special flags
            if transaction_data.get('Is_Weekend'):
                self.real_time_stats['weekend_transactions'] += 1
            
            if transaction_data.get('Is_Holiday'):
                self.real_time_stats['holiday_transactions'] += 1
            
            if transaction_data.get('Regulatory_Reporting_Required'):
                self.real_time_stats['regulatory_transactions'] += 1
            
            if status == 'FAILED':
                self.real_time_stats['failed_transactions'] += 1
            
            if transaction_data.get('AML_Flag'):
                self.real_time_stats['aml_flagged_transactions'] += 1
            
            if transaction_data.get('STR_Flag'):
                self.real_time_stats['suspicious_transactions'] += 1
            
            # Add to recent transactions for monitoring
            self.recent_transactions.append({
                'timestamp': datetime.now(),
                'txn_ref': transaction_data.get('Transaction_Ref_No', ''),  # Protected field - always valid
                'customer_id': transaction_data.get('Customer_ID', ''),
                'customer_name': transaction_data.get('Customer_Name', ''),
                'amount': amount,
                'status': status,
                'risk_level': risk_level,
                'customer_segment': segment,
                'account_type': account_type,
                'sequence_number': transaction_data.get('Transaction_Sequence_Number', 0),
                'opening_balance': transaction_data.get('Opening_Balance', 0),
                'closing_balance': transaction_data.get('Closing_Balance', 0),
                'sender_bank_name': transaction_data.get('Sender_Bank_Name', 'Unknown'),
                'receiver_bank_name': transaction_data.get('Receiver_Bank_Name', 'Unknown')
            })
            
        except Exception as e:
            # Continue processing even if stats update fails
            pass

    def real_time_dashboard(self):
        """Real-time dashboard that prints analytics every 30 seconds"""
        while self.dashboard_active:
            try:
                time.sleep(30)  # Update every 30 seconds
                if self.real_time_stats['total_transactions'] > 0:
                    # Dashboard content removed for silent operation
                    pass
            except Exception as e:
                pass

    def write_to_csv(self, transaction_data):
        """Write transaction data to CSV file with comprehensive headers"""
        try:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.exists(self.csv_filename)
            
            # Ensure backend directory exists
            os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
            
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                
                # Write headers if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Prepare row data - ensure all headers have values
                row_data = {}
                for header in self.csv_headers:
                    row_data[header] = transaction_data.get(header, None)
                
                writer.writerow(row_data)
                
        except Exception as e:
            pass  # Silently handle CSV write errors

    def process_transaction(self, transaction_data):
        """Process individual transaction with comprehensive analysis including bank statement format"""
        try:
            self.transaction_count += 1
            
            # Write to CSV
            self.write_to_csv(transaction_data)
            
            # Detect anomalies (enhanced for bank statement format with protected field validation)
            anomalies = self.detect_comprehensive_anomalies(transaction_data)
            if anomalies:
                self.records_with_anomalies += 1
                for anomaly in anomalies:
                    self.anomaly_stats[anomaly] += 1
            
            # Update real-time statistics (enhanced with bank statement metrics)
            self.update_real_time_stats(transaction_data)
            
            # Silent processing - no print statements
            
        except Exception as e:
            pass  # Silently handle processing errors

    def generate_customer_bank_statements(self, df):
        """Generate individual customer bank statement previews"""
        try:
            # Clean data first - remove anomaly characters from string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if col not in ['Transaction_Ref_No', 'Customer_ID', 'Account_Number', 'Customer_Name', 'UTR_Number', 'Cheque_Number', 'Statement_Number']:  # Skip protected fields
                    df[col] = df[col].astype(str).str.replace('@#$%^&*()', '', regex=False)
            
            # Convert dates and amounts
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['Value_Date'] = pd.to_datetime(df['Value_Date'], errors='coerce')
            df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')
            df['Opening_Balance'] = pd.to_numeric(df['Opening_Balance'], errors='coerce')
            df['Closing_Balance'] = pd.to_numeric(df['Closing_Balance'], errors='coerce')
            df['Transaction_Sequence_Number'] = pd.to_numeric(df['Transaction_Sequence_Number'], errors='coerce')
            
            # Filter valid data
            valid_df = df[(df['Transaction_Date'].notna()) & (df['Transaction_Amount'].notna()) & (df['Transaction_Amount'] > 0)]
            
            # Get top 3 most active customers for bank statement preview
            customer_freq = valid_df['Customer_ID'].value_counts().head(3)
            
            # Process customers silently for statement readiness verification
            statement_ready_customers = 0
            for customer_id, txn_count in customer_freq.items():
                customer_data = valid_df[valid_df['Customer_ID'] == customer_id].copy()
                customer_data = customer_data.sort_values(['Transaction_Date', 'Transaction_Sequence_Number'])
                
                if len(customer_data) > 0:
                    statement_ready_customers += 1
            
            return statement_ready_customers
        
        except Exception as e:
            return 0

    def generate_customer_frequency_reports(self, df):
        """Generate comprehensive customer frequency and behavior reports (enhanced for bank statements)"""
        try:
            # Clean data first - remove anomaly characters from string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if col not in ['Transaction_Ref_No', 'Customer_ID', 'Account_Number', 'Customer_Name', 'UTR_Number', 'Cheque_Number', 'Statement_Number']:  # Skip protected fields
                    df[col] = df[col].astype(str).str.replace('@#$%^&*()', '', regex=False)
            
            # Convert Transaction_Date to datetime for analysis
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')
            df['Transaction_Sequence_Number'] = pd.to_numeric(df['Transaction_Sequence_Number'], errors='coerce')
            df['Opening_Balance'] = pd.to_numeric(df['Opening_Balance'], errors='coerce')
            df['Closing_Balance'] = pd.to_numeric(df['Closing_Balance'], errors='coerce')
            
            # Filter valid data
            valid_df = df[(df['Transaction_Date'].notna()) & (df['Transaction_Amount'].notna()) & (df['Transaction_Amount'] > 0)]
            
            # Customer transaction frequency analysis
            customer_freq = valid_df['Customer_ID'].value_counts()
            customer_stats = pd.DataFrame({
                'Customer_ID': customer_freq.index,
                'Transaction_Count': customer_freq.values
            })
            
            # Add customer details (enhanced for bank statements)
            customer_details = valid_df.groupby('Customer_ID').agg({
                'Customer_Name': 'first',
                'Account_Number': 'first',
                'Account_Type': 'first',
                'Account_Type_Name': 'first',
                'Customer_Segment': 'first',
                'Customer_Type': 'first',
                'Transaction_Amount': ['sum', 'mean', 'median'],
                'Transaction_Date': ['min', 'max', 'nunique'],
                'Transaction_Status': lambda x: (x == 'SUCCESS').sum(),
                'Risk_Score': 'mean',
                'Closing_Balance': 'last',
                'Opening_Balance': 'first',
                'Statement_Number': 'first',
                'Statement_Period_Start': 'first',
                'Statement_Period_End': 'first',
                'Transaction_Sequence_Number': 'max',
                'UTR_Number': lambda x: x.notna().sum(),
                'Cheque_Number': lambda x: x.notna().sum(),
                'Sender_Bank_Name': 'first'
            }).round(2)
            
            # Flatten column names
            customer_details.columns = [
                'Customer_Name', 'Account_Number', 'Account_Type', 'Account_Type_Name', 'Customer_Segment', 'Customer_Type',
                'Total_Amount', 'Avg_Amount', 'Median_Amount', 'First_Transaction', 'Last_Transaction',
                'Active_Days', 'Successful_Transactions', 'Avg_Risk_Score', 'Current_Balance', 'Initial_Balance',
                'Statement_Number', 'Statement_Start', 'Statement_End', 'Max_Sequence_Number',
                'UTR_Count', 'Cheque_Count', 'Sender_Bank_Name'
            ]
            
            # Merge frequency with details
            customer_analysis = customer_stats.merge(customer_details, left_on='Customer_ID', right_index=True, how='left')
            customer_analysis['Success_Rate'] = (customer_analysis['Successful_Transactions'] / customer_analysis['Transaction_Count'] * 100).round(1)
            customer_analysis['Days_Active_Period'] = (customer_analysis['Last_Transaction'] - customer_analysis['First_Transaction']).dt.days + 1
            customer_analysis['Avg_Transactions_Per_Day'] = (customer_analysis['Transaction_Count'] / customer_analysis['Days_Active_Period']).round(2)
            customer_analysis['Balance_Change'] = customer_analysis['Current_Balance'] - customer_analysis['Initial_Balance']
            
            return len(customer_analysis)
        
        except Exception as e:
            return 0

    def generate_booking_reports(self, df):
        """Generate comprehensive booking reports for last month vs current month including bank statement metrics"""
        try:
            # Clean data first - remove anomaly characters from string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if col not in ['Transaction_Ref_No', 'Customer_ID', 'Account_Number', 'Customer_Name', 'UTR_Number', 'Cheque_Number', 'Statement_Number']:  # Skip protected fields
                    df[col] = df[col].astype(str).str.replace('@#$%^&*()', '', regex=False)
            
            # Convert Transaction_Date to datetime for analysis
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
            df['Value_Date'] = pd.to_datetime(df['Value_Date'], errors='coerce')
            df['Transaction_Amount'] = pd.to_numeric(df['Transaction_Amount'], errors='coerce')
            df['Opening_Balance'] = pd.to_numeric(df['Opening_Balance'], errors='coerce')
            df['Closing_Balance'] = pd.to_numeric(df['Closing_Balance'], errors='coerce')
            
            # Filter valid data
            valid_df = df[(df['Transaction_Date'].notna()) & (df['Transaction_Amount'].notna()) & (df['Transaction_Amount'] > 0)]
            
            # Month-wise analysis (enhanced with bank statement metrics)
            month_analysis = valid_df.groupby('Month_Category').agg({
                'Transaction_Ref_No': 'count',
                'Transaction_Amount': ['sum', 'mean', 'median', 'min', 'max'],
                'Customer_ID': 'nunique',
                'Account_Number': 'nunique',
                'Risk_Score': 'mean',
                'UTR_Number': lambda x: x.notna().sum(),
                'Cheque_Number': lambda x: x.notna().sum(),
                'Opening_Balance': 'mean',
                'Closing_Balance': 'mean',
                'Transaction_Sequence_Number': 'max',
                'Sender_Bank_Name': 'nunique'
            }).round(2)
            
            return len(month_analysis)
        
        except Exception as e:
            return 0

    def consume_and_analyze(self):
        """Main consumer loop with comprehensive analysis including bank statement format"""
        # Start real-time dashboard
        self.dashboard_thread.start()
        
        try:
            for message in self.consumer:
                transaction_data = message.value
                self.process_transaction(transaction_data)
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            pass
        finally:
            self.dashboard_active = False
            self.consumer.close()
            
            # Generate final comprehensive reports
            self.generate_final_reports()

    def generate_final_reports(self):
        """Generate comprehensive final reports including bank statement analysis"""
        # Load and analyze CSV data for comprehensive reports
        try:
            if os.path.exists(self.csv_filename):
                df = pd.read_csv(self.csv_filename)
                
                # Generate all comprehensive reports silently
                statement_ready_customers = self.generate_customer_bank_statements(df)
                customer_analysis_count = self.generate_customer_frequency_reports(df)
                booking_analysis_count = self.generate_booking_reports(df)
                
                # Silent completion
                return True
                
            else:
                return False
                
        except Exception as e:
            return False

if __name__ == "__main__":
    consumer = BankingAnalyticsConsumer()
    consumer.consume_and_analyze()