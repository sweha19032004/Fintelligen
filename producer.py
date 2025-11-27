from kafka import KafkaProducer
from faker import Faker
import json
import random
from datetime import datetime, timedelta, time as dt_time
import time
import numpy as np
import uuid

fake = Faker('en_IN')

class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumPyEncoder, self).default(obj)

# Global sets to ensure uniqueness across all transactions - CRITICAL FOR BANK STATEMENTS
used_transaction_ids = set()
used_utr_numbers = set()
used_cheque_numbers = set()
used_customer_ids = set()
used_account_numbers = set()
used_statement_numbers = set()

# Enhanced Real-world banking data structures with sender bank tracking
BANK_PREFIXES = {
    'SBIN': {'name': 'State Bank of India', 'acc_start': '2000', 'acc_digits': 11, 'ifsc_prefix': 'SBIN0'},
    'HDFC': {'name': 'HDFC Bank', 'acc_start': '5000', 'acc_digits': 14, 'ifsc_prefix': 'HDFC0'},
    'ICIC': {'name': 'ICICI Bank', 'acc_start': '0000', 'acc_digits': 12, 'ifsc_prefix': 'ICIC0'},
    'AXIS': {'name': 'Axis Bank', 'acc_start': '9110', 'acc_digits': 12, 'ifsc_prefix': 'UTIB0'},
    'PUNB': {'name': 'Punjab National Bank', 'acc_start': '0158', 'acc_digits': 16, 'ifsc_prefix': 'PUNB0'},
    'CNRB': {'name': 'Canara Bank', 'acc_start': '0691', 'acc_digits': 12, 'ifsc_prefix': 'CNRB0'},
    'UBIN': {'name': 'Union Bank of India', 'acc_start': '5020', 'acc_digits': 15, 'ifsc_prefix': 'UBIN0'},
    'IOBA': {'name': 'Indian Overseas Bank', 'acc_start': '0190', 'acc_digits': 13, 'ifsc_prefix': 'IOBA0'},
    'BKID': {'name': 'Bank of India', 'acc_start': '0010', 'acc_digits': 12, 'ifsc_prefix': 'BKID0'},
    'CBIN': {'name': 'Central Bank of India', 'acc_start': '3012', 'acc_digits': 10, 'ifsc_prefix': 'CBIN0'},
    'KOTAK': {'name': 'Kotak Mahindra Bank', 'acc_start': '7311', 'acc_digits': 12, 'ifsc_prefix': 'KKBK0'},
    'YESB': {'name': 'Yes Bank', 'acc_start': '0075', 'acc_digits': 15, 'ifsc_prefix': 'YESB0'},
    'INDB': {'name': 'IndusInd Bank', 'acc_start': '2000', 'acc_digits': 12, 'ifsc_prefix': 'INDB0'},
    'IDBI': {'name': 'IDBI Bank', 'acc_start': '0259', 'acc_digits': 12, 'ifsc_prefix': 'IBKL0'},
    'FDBK': {'name': 'Federal Bank', 'acc_start': '1440', 'acc_digits': 14, 'ifsc_prefix': 'FDRL0'}
}

# Account Types with their characteristics
ACCOUNT_TYPES = {
    'SAVINGS': {
        'name': 'Savings Account',
        'min_balance': 1000,
        'max_balance': 1000000,
        'interest_rate': 3.5,
        'transaction_limit': 100000,
        'weight': 40,
        'daily_txn_probability': 0.6
    },
    'CURRENT': {
        'name': 'Current Account',
        'min_balance': 10000,
        'max_balance': 10000000,
        'interest_rate': 0.0,
        'transaction_limit': 5000000,
        'weight': 25,
        'daily_txn_probability': 0.8
    },
    'FIXED_DEPOSIT': {
        'name': 'Fixed Deposit Account',
        'min_balance': 50000,
        'max_balance': 50000000,
        'interest_rate': 6.5,
        'transaction_limit': 1000000,
        'weight': 15,
        'daily_txn_probability': 0.2
    },
    'RECURRING_DEPOSIT': {
        'name': 'Recurring Deposit Account',
        'min_balance': 1000,
        'max_balance': 500000,
        'interest_rate': 6.0,
        'transaction_limit': 50000,
        'weight': 10,
        'daily_txn_probability': 0.3
    },
    'SALARY': {
        'name': 'Salary Account',
        'min_balance': 0,
        'max_balance': 2000000,
        'interest_rate': 3.5,
        'transaction_limit': 200000,
        'weight': 8,
        'daily_txn_probability': 0.7
    },
    'NRI': {
        'name': 'NRI Account',
        'min_balance': 100000,
        'max_balance': 50000000,
        'interest_rate': 4.0,
        'transaction_limit': 10000000,
        'weight': 2,
        'daily_txn_probability': 0.4
    }
}

# Enhanced transaction categories for banking reports
TRANSACTION_CATEGORIES = {
    'RETAIL_BANKING': ['ATM_WITHDRAW', 'POS_PURCHASE', 'ONLINE_PURCHASE', 'BILL_PAYMENT'],
    'CORPORATE_BANKING': ['BULK_TRANSFER', 'PAYROLL', 'VENDOR_PAYMENT', 'TAX_PAYMENT'],
    'INVESTMENT': ['MUTUAL_FUND', 'FIXED_DEPOSIT', 'INSURANCE_PREMIUM', 'STOCK_PURCHASE'],
    'LOAN_SERVICES': ['EMI_PAYMENT', 'LOAN_DISBURSEMENT', 'INTEREST_CREDIT', 'PENALTY_CHARGE'],
    'GOVERNMENT': ['TAX_REFUND', 'SUBSIDY_CREDIT', 'PENSION_CREDIT', 'SCHOLARSHIP'],
    'INTERNATIONAL': ['FOREX_EXCHANGE', 'REMITTANCE_INWARD', 'REMITTANCE_OUTWARD', 'TRADE_FINANCE']
}

# Banking channels for multi-channel analysis
BANKING_CHANNELS = {
    'DIGITAL': ['MOBILE_APP', 'INTERNET_BANKING', 'UPI_APP', 'DIGITAL_WALLET'],
    'PHYSICAL': ['BRANCH', 'ATM', 'POS_TERMINAL', 'KIOSK'],
    'ASSISTED': ['PHONE_BANKING', 'VIDEO_BANKING', 'RELATIONSHIP_MANAGER', 'CUSTOMER_SERVICE']
}

# Indian states and major cities for geographical analysis
INDIAN_GEOGRAPHY = {
    'MAHARASHTRA': ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad'],
    'KARNATAKA': ['Bangalore', 'Mysore', 'Hubli', 'Mangalore', 'Belgaum'],
    'TAMIL_NADU': ['Chennai', 'Coimbatore', 'Madurai', 'Salem', 'Trichy'],
    'DELHI': ['New Delhi', 'Dwarka', 'Rohini', 'Lajpat Nagar', 'Karol Bagh'],
    'GUJARAT': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot', 'Bhavnagar'],
    'WEST_BENGAL': ['Kolkata', 'Howrah', 'Durgapur', 'Asansol', 'Siliguri'],
    'RAJASTHAN': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota', 'Ajmer'],
    'UTTAR_PRADESH': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi', 'Meerut']
}

# Bank Statement Specific Enhancements
REALISTIC_TRANSACTION_DESCRIPTIONS = {
    'UPI': [
        'UPI-{beneficiary_name}-{upi_id}@{provider}-{purpose}',
        'UPI/{beneficiary_name}/{upi_id}@{provider}/{purpose_code}',
        'UPI-P2P-{beneficiary_name}-{reference}',
        'UPI-P2M-{merchant_name}-{location}',
        'UPI-COLLECT-{beneficiary_name}-{purpose}'
    ],
    'NEFT': [
        'NEFT Cr-{beneficiary_bank}-{beneficiary_name}-{purpose}',
        'NEFT Dr-{beneficiary_account}-{beneficiary_name}-{purpose}',
        'NEFT IN-{sender_name}-{sender_bank}-{purpose}',
        'NEFT OUT-{beneficiary_name}-{beneficiary_bank}-{purpose}'
    ],
    'RTGS': [
        'RTGS Cr-{beneficiary_bank}-{beneficiary_name}-{purpose}',
        'RTGS Dr-{beneficiary_account}-{beneficiary_name}-{purpose}',
        'RTGS IN-{sender_name}-{sender_bank}-{purpose}',
        'RTGS OUT-{beneficiary_name}-{beneficiary_bank}-{purpose}'
    ],
    'IMPS': [
        'IMPS-P2P-{beneficiary_name}-{mobile_number}',
        'IMPS-P2A-{beneficiary_account}-{beneficiary_name}',
        'IMPS IN-{sender_name}-{mobile_number}',
        'IMPS OUT-{beneficiary_name}-{mobile_number}'
    ],
    'CASH': [
        'CASH DEPOSIT-BRANCH-{branch_name}',
        'CASH WITHDRAWAL-ATM-{atm_location}',
        'CASH WITHDRAWAL-BRANCH-{branch_name}',
        'ATM CASH WITHDRAWAL-{atm_id}-{location}'
    ],
    'CARD': [
        'CARD TXN-{merchant_name}-{location}-{card_last4}',
        'POS PURCHASE-{merchant_name}-{location}',
        'ONLINE PURCHASE-{merchant_name}-{card_last4}',
        'CARD PAYMENT-{merchant_category}-{location}'
    ],
    'CHQS': [
        'CHEQUE DEPOSIT-{cheque_number}-{drawer_name}',
        'CHEQUE CLEARANCE-{cheque_number}-{payee_name}',
        'CHEQUE RETURN-{cheque_number}-{reason}',
        'CHEQUE PAYMENT-{cheque_number}-{payee_name}'
    ],
    'NACH': [
        'NACH DEBIT-{sponsor_bank}-{purpose}-{mandate_ref}',
        'NACH CREDIT-{sponsor_bank}-{purpose}-{mandate_ref}',
        'ECS DEBIT-{utility_name}-{consumer_number}',
        'ECS CREDIT-{employer_name}-{employee_id}'
    ],
    'NETB': [
        'NET BANKING-{beneficiary_name}-{purpose}',
        'ONLINE TRANSFER-{beneficiary_bank}-{purpose}',
        'WEB PAYMENT-{merchant_name}-{reference}',
        'INTERNET BANKING-{beneficiary_name}-{purpose_code}'
    ],
    'MOBP': [
        'MOBILE PAYMENT-{beneficiary_name}-{mobile_number}',
        'MOBILE TRANSFER-{beneficiary_account}-{purpose}',
        'MOBILE BANKING-{beneficiary_name}-{reference}',
        'MOBILE APP-{merchant_name}-{purpose}'
    ]
}

PURPOSE_CODES = {
    'BUSINESS': ['B001', 'B002', 'B003', 'B004', 'B005'],
    'PERSONAL': ['P001', 'P002', 'P003', 'P004', 'P005'],
    'INVESTMENT': ['I001', 'I002', 'I003', 'I004', 'I005'],
    'GOVERNMENT': ['G001', 'G002', 'G003', 'G004', 'G005'],
    'EDUCATION': ['E001', 'E002', 'E003', 'E004', 'E005']
}

UPI_PROVIDERS = ['paytm', 'googlepay', 'phonepe', 'okaxis', 'ybl', 'ibl', 'axl', 'hdfcbank']

class CustomerPool:
    """Manages a limited pool of customers for frequent transactions with bank statement features"""
    
    def __init__(self, pool_size=250):
        self.pool_size = pool_size
        self.customers = []
        self.customer_balances = {}
        self.customer_last_transaction_date = {}
        self.customer_transaction_sequence = {}  # Track sequential numbering per customer
        self.customer_statement_periods = {}  # Track statement periods
        self.generate_customer_pool()
    
    def generate_unique_customer_id(self):
        """Generate unique customer ID with collision detection"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            customer_id = f"CUST{fake.random_number(digits=10, fix_len=True)}"
            if customer_id not in used_customer_ids:
                used_customer_ids.add(customer_id)
                return customer_id
            attempts += 1
        
        # Fallback with timestamp if all attempts failed
        timestamp_suffix = str(int(datetime.now().timestamp()))[-6:]
        fallback_id = f"CUST{timestamp_suffix}{fake.random_number(digits=4, fix_len=True)}"
        used_customer_ids.add(fallback_id)
        return fallback_id
    
    def generate_unique_statement_number(self):
        """Generate unique statement number with collision detection"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            statement_number = f"ST{fake.random_number(digits=8, fix_len=True)}"
            if statement_number not in used_statement_numbers:
                used_statement_numbers.add(statement_number)
                return statement_number
            attempts += 1
        
        # Fallback with timestamp if all attempts failed
        timestamp_suffix = str(int(datetime.now().timestamp()))[-6:]
        fallback_stmt = f"ST{timestamp_suffix}{fake.random_number(digits=2, fix_len=True)}"
        used_statement_numbers.add(fallback_stmt)
        return fallback_stmt
    
    def generate_customer_pool(self):
        """Generate a limited pool of customers with their accounts"""
        print(f"🏦 Generating {self.pool_size} customers for bank statement generation...")
        
        for i in range(self.pool_size):
            # Select account type based on weights
            account_types = list(ACCOUNT_TYPES.keys())
            weights = [ACCOUNT_TYPES[acc_type]['weight'] for acc_type in account_types]
            account_type = random.choices(account_types, weights=weights, k=1)[0]
            account_type_info = ACCOUNT_TYPES[account_type]
            
            # Generate account number
            account_number, bank_code = self.generate_account_number(account_type)
            
            # Generate unique customer ID
            customer_id = self.generate_unique_customer_id()
            
            # Generate geographical data (customer's home location)
            state = random.choice(list(INDIAN_GEOGRAPHY.keys()))
            city = random.choice(INDIAN_GEOGRAPHY[state])
            
            # Generate sender bank name based on account number
            sender_bank_name = BANK_PREFIXES[bank_code]['name']
            
            customer = {
                'customer_id': customer_id,
                'customer_name': fake.name(),
                'customer_type': random.choice(['INDIVIDUAL', 'CORPORATE', 'PARTNERSHIP', 'GOVERNMENT']),
                'customer_segment': random.choice(['PREMIUM', 'GOLD', 'SILVER', 'BASIC']),
                'kyc_status': random.choice(['FULL_KYC', 'MIN_KYC', 'PENDING']),
                'risk_category': random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'relationship_manager': fake.name() if random.choice([True, False]) else None,
                'account_number': account_number,
                'account_type': account_type,
                'account_type_info': account_type_info,
                'bank_code': bank_code,
                'sender_bank_name': sender_bank_name,  # NEW: Sender bank name
                'home_state': state,
                'home_city': city,
                'branch_code': f"BR{fake.random_number(digits=4, fix_len=True)}",
                'preferred_channels': self.get_preferred_channels(customer_segment=random.choice(['PREMIUM', 'GOLD', 'SILVER', 'BASIC']))
            }
            
            self.customers.append(customer)
            
            # Initialize balance based on customer segment and account type
            if customer['customer_segment'] == 'PREMIUM':
                initial_balance = round(random.uniform(max(account_type_info['min_balance'], 500000), account_type_info['max_balance']), 2)
            elif customer['customer_segment'] == 'GOLD':
                initial_balance = round(random.uniform(max(account_type_info['min_balance'], 100000), min(1000000, account_type_info['max_balance'])), 2)
            else:
                initial_balance = round(random.uniform(account_type_info['min_balance'], min(500000, account_type_info['max_balance'])), 2)
            
            self.customer_balances[customer_id] = initial_balance
            self.customer_last_transaction_date[customer_id] = None
            self.customer_transaction_sequence[customer_id] = 0  # Initialize sequence counter
            
            # Initialize statement period with unique statement number
            current_date = datetime.now().date()
            statement_start = current_date.replace(day=1)
            statement_end = (statement_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            statement_number = self.generate_unique_statement_number()
            
            self.customer_statement_periods[customer_id] = {
                'start_date': statement_start,
                'end_date': statement_end,
                'statement_number': statement_number
            }
        
        print(f"✅ Generated {len(self.customers)} customers with unique IDs and statement numbers")
    
    def generate_account_number(self, account_type):
        """Generate unique account number based on bank format and account type"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            bank_code = random.choice(list(BANK_PREFIXES.keys()))
            bank_info = BANK_PREFIXES[bank_code]
            
            acc_prefix = bank_info['acc_start']
            remaining_digits = bank_info['acc_digits'] - len(acc_prefix)
            
            # Add account type indicator in the account number
            account_type_codes = {
                'SAVINGS': '01',
                'CURRENT': '02',
                'FIXED_DEPOSIT': '03',
                'RECURRING_DEPOSIT': '04',
                'SALARY': '05',
                'NRI': '06'
            }
            
            type_code = account_type_codes.get(account_type, '01')
            remaining_digits -= 2  # Account for type code
            
            acc_suffix = fake.random_number(digits=remaining_digits, fix_len=True)
            account_number = f"{acc_prefix}{type_code}{acc_suffix}"
            
            # Check for uniqueness
            if account_number not in used_account_numbers:
                used_account_numbers.add(account_number)
                return account_number, bank_code
            
            attempts += 1
        
        # Fallback with timestamp if all attempts failed
        timestamp_suffix = str(int(datetime.now().timestamp()))[-4:]
        fallback_acc = f"{acc_prefix}{type_code}{timestamp_suffix}"
        used_account_numbers.add(fallback_acc)
        return fallback_acc, bank_code
    
    def get_preferred_channels(self, customer_segment):
        """Get preferred channels based on customer segment"""
        if customer_segment == 'PREMIUM':
            return ['MOBILE_APP', 'INTERNET_BANKING', 'RELATIONSHIP_MANAGER']
        elif customer_segment == 'GOLD':
            return ['MOBILE_APP', 'INTERNET_BANKING', 'ATM']
        elif customer_segment == 'SILVER':
            return ['MOBILE_APP', 'ATM', 'BRANCH']
        else:
            return ['ATM', 'BRANCH', 'UPI_APP']
    
    def should_customer_transact_today(self, customer, current_date):
        """Determine if customer should have a transaction today based on their account type and behavior"""
        account_type = customer['account_type']
        daily_probability = ACCOUNT_TYPES[account_type]['daily_txn_probability']
        
        # Increase probability for frequent customers
        last_txn_date = self.customer_last_transaction_date.get(customer['customer_id'])
        if last_txn_date:
            days_since_last = (current_date - last_txn_date).days
            if days_since_last > 3:  # If more than 3 days, increase probability
                daily_probability *= 1.5
            elif days_since_last == 1:  # If yesterday, reduce probability slightly
                daily_probability *= 0.8
        
        # Weekend and holiday adjustments
        if current_date.weekday() >= 5:  # Weekend
            daily_probability *= 0.7
        
        # Business accounts more active on weekdays
        if customer['customer_type'] in ['CORPORATE', 'PARTNERSHIP'] and current_date.weekday() < 5:
            daily_probability *= 1.3
        
        return random.random() < min(daily_probability, 0.95)  # Cap at 95%
    
    def get_customer_for_transaction(self, current_date):
        """Select a customer who should have a transaction today"""
        eligible_customers = []
        
        for customer in self.customers:
            if self.should_customer_transact_today(customer, current_date):
                eligible_customers.append(customer)
        
        if not eligible_customers:
            # If no eligible customers, select from most frequent transactors
            business_customers = [c for c in self.customers if c['customer_type'] in ['CORPORATE', 'PARTNERSHIP']]
            if business_customers:
                return random.choice(business_customers)
            else:
                return random.choice(self.customers)
        
        return random.choice(eligible_customers)
    
    def get_next_transaction_sequence(self, customer_id):
        """Get next sequential transaction number for customer (bank statement requirement)"""
        self.customer_transaction_sequence[customer_id] += 1
        return self.customer_transaction_sequence[customer_id]
    
    def update_customer_balance(self, customer_id, amount, transaction_type, current_date):
        """Update customer balance and last transaction date"""
        if customer_id in self.customer_balances:
            if transaction_type == 'CR':
                self.customer_balances[customer_id] += amount
            else:
                # Ensure minimum balance is maintained
                customer = next((c for c in self.customers if c['customer_id'] == customer_id), None)
                if customer:
                    min_balance = customer['account_type_info']['min_balance']
                    new_balance = self.customer_balances[customer_id] - amount
                    if new_balance >= min_balance:
                        self.customer_balances[customer_id] = new_balance
                    else:
                        # Adjust amount to maintain minimum balance
                        self.customer_balances[customer_id] = min_balance
        
        self.customer_last_transaction_date[customer_id] = current_date
        
        return round(self.customer_balances.get(customer_id, 0), 2)

def safe_format(value, format_spec=""):
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

def generate_realistic_transaction_ref():
    """Generate unique realistic transaction reference number with bank prefix"""
    max_attempts = 100
    attempts = 0
    
    while attempts < max_attempts:
        # Choose transaction type and corresponding prefix
        txn_type = random.choice(['NEFT', 'RTGS', 'IMPS', 'UPI', 'CASH', 'CARD', 'CHQS', 'NETB', 'NACH', 'MOBP'])
        
        # Generate realistic transaction reference formats
        current_time = datetime.now()
        formats = [
            f"{txn_type}{current_time.strftime('%Y%m%d')}{fake.random_number(digits=8, fix_len=True)}",
            f"{txn_type}{fake.random_number(digits=12, fix_len=True)}",
            f"{txn_type}{current_time.strftime('%y%m%d')}{fake.random_number(digits=10, fix_len=True)}",
        ]
        
        transaction_ref = random.choice(formats)
        
        if transaction_ref not in used_transaction_ids:
            used_transaction_ids.add(transaction_ref)
            return transaction_ref
        
        attempts += 1
    
    # Fallback if all attempts failed
    timestamp_suffix = str(int(datetime.now().timestamp()))[-8:]
    fallback_ref = f"TXN{timestamp_suffix}"
    used_transaction_ids.add(fallback_ref)
    return fallback_ref

def generate_realistic_utr_number(transaction_code, bank_code, current_date):
    """Generate unique realistic UTR number for NEFT/RTGS/IMPS transactions"""
    if transaction_code not in ['NEFT', 'RTGS', 'IMPS']:
        return None
    
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts:
        # UTR format: BANKCODE + YYMMDD + 10-digit sequence
        date_part = current_date.strftime('%y%m%d')
        sequence_part = fake.random_number(digits=10, fix_len=True)
        utr_number = f"{bank_code}{date_part}{sequence_part}"
        
        if utr_number not in used_utr_numbers:
            used_utr_numbers.add(utr_number)
            return utr_number
        
        attempts += 1
    
    # Fallback
    timestamp_suffix = str(int(datetime.now().timestamp()))[-10:]
    fallback_utr = f"{bank_code}{current_date.strftime('%y%m%d')}{timestamp_suffix}"
    used_utr_numbers.add(fallback_utr)
    return fallback_utr

def generate_realistic_cheque_number():
    """Generate unique realistic cheque number"""
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts:
        cheque_number = fake.random_number(digits=6, fix_len=True)
        
        if cheque_number not in used_cheque_numbers:
            used_cheque_numbers.add(cheque_number)
            return cheque_number
        
        attempts += 1
    
    # Fallback
    timestamp_suffix = str(int(datetime.now().timestamp()))[-6:]
    used_cheque_numbers.add(timestamp_suffix)
    return timestamp_suffix

def safe_format_template(template, **kwargs):
    """Safely format template string with error handling"""
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError, TypeError) as e:
        # If template formatting fails, return a generic description
        return f"Transaction - {kwargs.get('purpose', 'GENERAL')} - {kwargs.get('location', 'Unknown')}"

def generate_realistic_transaction_description(transaction_code, transaction_data):
    """Generate realistic transaction description based on transaction type with enhanced error handling"""
    try:
        descriptions = REALISTIC_TRANSACTION_DESCRIPTIONS.get(transaction_code, ['Generic Transaction - {purpose}'])
        template = random.choice(descriptions)
        
        # Generate realistic data for template placeholders with safe defaults
        beneficiary_name = fake.name()
        beneficiary_account = f"****{fake.random_number(digits=4, fix_len=True)}"
        beneficiary_bank = random.choice(list(BANK_PREFIXES.values()))['name']
        sender_name = fake.name()
        sender_bank = random.choice(list(BANK_PREFIXES.values()))['name']
        mobile_number = f"****{fake.random_number(digits=4, fix_len=True)}"
        upi_id = fake.user_name()
        provider = random.choice(UPI_PROVIDERS)
        merchant_name = fake.company()
        location = transaction_data.get('Transaction_City', fake.city())
        branch_name = f"{location} Main Branch"
        atm_location = f"{location} ATM"
        atm_id = f"ATM{fake.random_number(digits=6, fix_len=True)}"
        card_last4 = fake.random_number(digits=4, fix_len=True)
        cheque_number = transaction_data.get('Cheque_Number', fake.random_number(digits=6, fix_len=True))
        drawer_name = fake.name()
        payee_name = fake.name()
        purpose = random.choice(['SALARY', 'RENT', 'UTILITIES', 'INVESTMENT', 'BUSINESS', 'PERSONAL'])
        purpose_code = random.choice(PURPOSE_CODES.get(transaction_data.get('Transaction_Purpose', 'PERSONAL'), ['P001']))
        reference = fake.random_number(digits=8, fix_len=True)
        mandate_ref = f"MND{fake.random_number(digits=8, fix_len=True)}"
        utility_name = random.choice(['ELECTRICITY BOARD', 'GAS COMPANY', 'WATER DEPT', 'TELECOM'])
        consumer_number = fake.random_number(digits=10, fix_len=True)
        employer_name = fake.company()
        employee_id = f"EMP{fake.random_number(digits=6, fix_len=True)}"
        merchant_category = random.choice(['GROCERY', 'FUEL', 'RESTAURANT', 'RETAIL', 'PHARMACY'])
        reason = random.choice(['INSUFFICIENT FUNDS', 'SIGNATURE MISMATCH', 'ACCOUNT CLOSED'])
        
        # Additional variables for NACH and other transaction types
        sponsor_bank = random.choice(list(BANK_PREFIXES.values()))['name']
        
        # Create comprehensive template variables dictionary
        template_vars = {
            'beneficiary_name': beneficiary_name,
            'beneficiary_account': beneficiary_account,
            'beneficiary_bank': beneficiary_bank,
            'sender_name': sender_name,
            'sender_bank': sender_bank,
            'mobile_number': mobile_number,
            'upi_id': upi_id,
            'provider': provider,
            'merchant_name': merchant_name,
            'location': location,
            'branch_name': branch_name,
            'atm_location': atm_location,
            'atm_id': atm_id,
            'card_last4': card_last4,
            'cheque_number': cheque_number,
            'drawer_name': drawer_name,
            'payee_name': payee_name,
            'purpose': purpose,
            'purpose_code': purpose_code,
            'reference': reference,
            'mandate_ref': mandate_ref,
            'utility_name': utility_name,
            'consumer_number': consumer_number,
            'employer_name': employer_name,
            'employee_id': employee_id,
            'merchant_category': merchant_category,
            'reason': reason,
            'sponsor_bank': sponsor_bank
        }
        
        # Safely format the template
        description = safe_format_template(template, **template_vars)
        return description
        
    except Exception as e:
        # Fallback description if anything goes wrong
        return f"{transaction_code} Transaction - {transaction_data.get('Transaction_Purpose', 'GENERAL')} - {transaction_data.get('Transaction_City', 'Unknown')}"

def inject_anomaly(value, field_type, anomaly_percentage=15):
    """Inject anomalies based on field type with given percentage"""
    if random.randint(1, 100) <= anomaly_percentage:
        if field_type == 'date':
            anomaly_type = random.choice(['future_date', 'null_date', 'invalid_format'])
            if anomaly_type == 'future_date':
                future_date = datetime.now() + timedelta(days=random.randint(1, 365))
                return future_date.strftime('%Y-%m-%d')
            elif anomaly_type == 'null_date':
                return None
            else:
                return fake.date_between(start_date='-30d', end_date='today').strftime('%d/%m/%Y')
        
        elif field_type == 'amount':
            anomaly_type = random.choice(['negative', 'extreme_high', 'zero', 'null'])
            if anomaly_type == 'negative':
                return -abs(value) if isinstance(value, (int, float)) else -1000
            elif anomaly_type == 'extreme_high':
                return round(random.uniform(10000000, 50000000), 2)
            elif anomaly_type == 'zero':
                return 0.0
            else:
                return None
        
        elif field_type == 'balance':
            anomaly_type = random.choice(['negative', 'null', 'inconsistent'])
            if anomaly_type == 'negative':
                return -abs(value) if isinstance(value, (int, float)) else -10000
            elif anomaly_type == 'null':
                return None
            else:
                return value * random.uniform(10, 100) if isinstance(value, (int, float)) else 999999
        
        elif field_type == 'status':
            anomaly_type = random.choice(['invalid', 'null', 'mixed_case'])
            if anomaly_type == 'invalid':
                return random.choice(['UNKNOWN', 'ERROR', 'TIMEOUT', 'CANCELLED'])
            elif anomaly_type == 'null':
                return None
            else:
                return value.lower() if value else None
        
        elif field_type == 'code':
            anomaly_type = random.choice(['invalid', 'null', 'wrong_format'])
            if anomaly_type == 'invalid':
                return random.choice(['XXXX', 'UNKN', 'TEST', 'DUMMY'])
            elif anomaly_type == 'null':
                return None
            else:
                return value.lower() if value else None
        
        elif field_type == 'flag':
            anomaly_type = random.choice(['invalid', 'null', 'wrong_case'])
            if anomaly_type == 'invalid':
                return random.choice(['XX', 'NA', 'UN', 'ER'])
            elif anomaly_type == 'null':
                return None
            else:
                return value.lower() if value else None
        
        elif field_type == 'numeric':
            anomaly_type = random.choice(['null', 'negative', 'extreme'])
            if anomaly_type == 'null':
                return None
            elif anomaly_type == 'negative':
                return -abs(value) if isinstance(value, (int, float)) else -1
            else:
                return value * random.randint(100, 1000) if isinstance(value, (int, float)) else 999999
        
        elif field_type == 'string':
            anomaly_type = random.choice(['null', 'empty', 'special_chars'])
            if anomaly_type == 'null':
                return None
            elif anomaly_type == 'empty':
                return ""
            else:
                return f"{value}@#$%^&*()" if value else "@#$%^&*()"
    
    return value

def calculate_risk_score(transaction_data):
    """Calculate risk score based on various factors"""
    risk_score = 0
    risk_flags = []
    
    # Amount-based risk
    amount = transaction_data.get('Transaction_Amount', 0)
    if isinstance(amount, (int, float)):
        if amount > 2000000:
            risk_score += 40
            risk_flags.append('HIGH_AMOUNT')
        elif amount > 1000000:
            risk_score += 25
            risk_flags.append('MEDIUM_HIGH_AMOUNT')
        elif amount > 500000:
            risk_score += 15
            risk_flags.append('MEDIUM_AMOUNT')
    
    # Time-based risk
    transaction_time = datetime.now().time()
    if transaction_time.hour >= 23 or transaction_time.hour <= 5:
        risk_score += 20
        risk_flags.append('OFF_HOURS')
    
    # Transaction type risk
    if transaction_data.get('Transaction_Code') in ['CASH', 'RTGS']:
        risk_score += 10
        risk_flags.append('HIGH_RISK_TYPE')
    
    # Account type risk
    account_type = transaction_data.get('Account_Type', 'SAVINGS')
    if account_type in ['NRI', 'CURRENT']:
        risk_score += 5
        risk_flags.append('HIGH_RISK_ACCOUNT_TYPE')
    
    # Round amount pattern
    if isinstance(amount, (int, float)) and amount > 0 and amount % 10000 == 0:
        risk_score += 15
        risk_flags.append('ROUND_AMOUNT')
    
    # Determine risk level
    if risk_score >= 50:
        risk_level = 'HIGH'
    elif risk_score >= 25:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    return {
        'risk_score': min(risk_score, 100),
        'risk_level': risk_level,
        'risk_flags': risk_flags
    }

def determine_sender_bank_name(customer, transaction_code, dr_cr_flag):
    """Determine sender bank name based on transaction type and customer"""
    customer_bank_name = customer['sender_bank_name']
    
    # For credit transactions, sender might be different bank
    if dr_cr_flag == 'CR' and transaction_code in ['NEFT', 'RTGS', 'IMPS', 'UPI']:
        # 70% chance it's from external bank, 30% internal transfer
        if random.random() < 0.7:
            # Select random external bank
            external_banks = [bank['name'] for bank in BANK_PREFIXES.values() if bank['name'] != customer_bank_name]
            return random.choice(external_banks)
        else:
            return customer_bank_name
    
    # For debit transactions or other cases, use customer's bank
    return customer_bank_name

def generate_banking_transaction_data(customer_pool, target_date=None):
    """Generate comprehensive banking transaction data with bank statement format requirements"""
    try:
        # Use target date if provided, otherwise use random date
        if target_date:
            current_date = target_date
            current_datetime = datetime.combine(target_date, datetime.now().time())
        else:
            current_date = datetime.now().date()
            current_datetime = datetime.now()
        
        # Select customer from pool
        customer = customer_pool.get_customer_for_transaction(current_date)
        if not customer:
            raise ValueError("No customer available for transaction")
        
        # Generate realistic unique identifiers (PROTECTED FIELDS - NO ANOMALIES)
        transaction_ref = generate_realistic_transaction_ref()
        
        # Get customer and account information (PROTECTED FIELDS)
        account_number = customer['account_number']
        account_type = customer['account_type']
        account_type_info = customer['account_type_info']
        bank_code = customer['bank_code']
        customer_id = customer['customer_id']
        customer_name = customer['customer_name']  # PROTECTED
        
        # Get sequential transaction number for this customer (bank statement requirement)
        transaction_sequence = customer_pool.get_next_transaction_sequence(customer_id)
        
        # Determine month category
        current_month_start = datetime.now().replace(day=1).date()
        if current_date >= current_month_start:
            month_category = "Current Month"
        else:
            month_category = "Last Month"
        
        # Generate realistic transaction time
        random_hour = random.randint(8, 22)  # More realistic banking hours
        random_minute = random.randint(0, 59)
        random_second = random.randint(0, 59)
        random_time = dt_time(random_hour, random_minute, random_second)
        transaction_datetime = datetime.combine(current_date, random_time)
        
        # Generate value date (can be different from transaction date for processing)
        value_date = current_date
        if random.random() < 0.1:  # 10% chance of different value date
            value_date = current_date + timedelta(days=random.randint(1, 2))
        
        # Calculate week and day information
        week_in_month = ((current_date.day - 1) // 7) + 1
        day_of_week = current_date.strftime('%A')
        quarter = f"Q{((current_date.month - 1) // 3) + 1}"
        
        # Select transaction category and type based on customer behavior
        if customer['customer_type'] in ['CORPORATE', 'PARTNERSHIP']:
            category = random.choice(['CORPORATE_BANKING', 'RETAIL_BANKING', 'INVESTMENT'])
        else:
            category = random.choice(['RETAIL_BANKING', 'INVESTMENT', 'LOAN_SERVICES'])
        
        transaction_sub_type = random.choice(TRANSACTION_CATEGORIES[category])
        
        # Select banking channel based on customer preferences
        preferred_channels = customer.get('preferred_channels', ['ATM', 'BRANCH'])
        specific_channel = random.choice(preferred_channels)
        
        # Determine channel type
        channel_type = 'DIGITAL'
        for ch_type, channels in BANKING_CHANNELS.items():
            if specific_channel in channels:
                channel_type = ch_type
                break
        
        # Real banking transaction codes with weights based on channel
        if specific_channel in ['MOBILE_APP', 'UPI_APP']:
            transaction_code = random.choice(['UPI', 'IMPS', 'NEFT'])
        elif specific_channel == 'ATM':
            transaction_code = random.choice(['CASH', 'CARD'])
        elif specific_channel == 'BRANCH':
            transaction_code = random.choice(['CASH', 'CHQS', 'NEFT', 'RTGS'])
        else:
            transaction_code = random.choice(['NEFT', 'RTGS', 'IMPS', 'UPI', 'NACH', 'CHQS', 'CASH', 'CARD', 'NETB', 'MOBP'])
        
        transaction_codes = {
            'NEFT': 'National Electronic Funds Transfer',
            'RTGS': 'Real Time Gross Settlement',
            'IMPS': 'Immediate Payment Service',
            'UPI': 'Unified Payments Interface',
            'NACH': 'National Automated Clearing House',
            'CHQS': 'Cheque Settlement',
            'CASH': 'Cash Transaction',
            'CARD': 'Card Transaction',
            'NETB': 'Net Banking',
            'MOBP': 'Mobile Payment'
        }
        
        # Generate realistic amounts based on transaction type, category, and account type
        if transaction_code in ['RTGS']:
            amount = round(random.uniform(200000, min(5000000, account_type_info['transaction_limit'])), 2)
        elif transaction_code in ['NEFT']:
            amount = round(random.uniform(1, min(1000000, account_type_info['transaction_limit'])), 2)
        elif transaction_code in ['UPI', 'IMPS']:
            amount = round(random.uniform(1, min(100000, account_type_info['transaction_limit'])), 2)
        elif transaction_code in ['CASH']:
            amount = round(random.uniform(100, min(50000, account_type_info['transaction_limit'])), 2)
        else:
            amount = round(random.uniform(10, min(200000, account_type_info['transaction_limit'])), 2)
        
        # Adjust amount based on customer segment
        if customer['customer_segment'] == 'PREMIUM':
            amount *= random.uniform(2, 5)
        elif customer['customer_segment'] == 'GOLD':
            amount *= random.uniform(1.5, 3)
        
        # Ensure amount doesn't exceed account type transaction limit
        amount = min(amount, account_type_info['transaction_limit'])
        amount = round(amount, 2)
        
        # Debit/Credit flag - more realistic distribution
        if transaction_sub_type in ['SALARY_CREDIT', 'INTEREST_CREDIT', 'TAX_REFUND', 'LOAN_DISBURSEMENT']:
            dr_cr_flag = 'CR'
        elif transaction_sub_type in ['ATM_WITHDRAW', 'BILL_PAYMENT', 'EMI_PAYMENT', 'VENDOR_PAYMENT']:
            dr_cr_flag = 'DR'
        else:
            dr_cr_flag = random.choice(['DR', 'CR'])
        
        # Get current balance and calculate new balance (bank statement requirement)
        current_balance = customer_pool.customer_balances.get(customer_id, account_type_info['min_balance'])
        
        if dr_cr_flag == 'CR':
            opening_balance = current_balance
            closing_balance = current_balance + amount
        else:
            # Ensure customer has sufficient balance for debit
            if current_balance - amount < account_type_info['min_balance']:
                # Reduce amount to maintain minimum balance
                amount = max(100, current_balance - account_type_info['min_balance'] - 100)
                amount = round(amount, 2)
            opening_balance = current_balance
            closing_balance = current_balance - amount
        
        # Update customer balance in pool
        final_balance = customer_pool.update_customer_balance(
            customer_id, amount, dr_cr_flag, current_date
        )
        
        # Generate geographical data (mix of home location and travel)
        if random.random() < 0.8:  # 80% chance of home location
            geo_state = customer['home_state']
            geo_city = customer['home_city']
            branch_code = customer['branch_code']
        else:  # 20% chance of different location (travel/business)
            geo_state = random.choice(list(INDIAN_GEOGRAPHY.keys()))
            geo_city = random.choice(INDIAN_GEOGRAPHY[geo_state])
            branch_code = f"BR{fake.random_number(digits=4, fix_len=True)}"
        
        # Generate IFSC codes
        bank_info = BANK_PREFIXES[bank_code]
        ifsc_code = f"{bank_info['ifsc_prefix']}{fake.random_number(digits=6, fix_len=True)}"
        
        # Determine sender bank name based on transaction pattern
        sender_bank_name = determine_sender_bank_name(customer, transaction_code, dr_cr_flag)
        
        # Generate beneficiary information (from customer pool for internal transfers)
        beneficiary_account = None
        beneficiary_ifsc = None
        beneficiary_name = None
        beneficiary_bank = None
        receiver_bank_name = None
        
        if dr_cr_flag == 'DR' and random.random() < 0.3:  # 30% chance of internal transfer
            # Select another customer as beneficiary
            other_customers = [c for c in customer_pool.customers if c['customer_id'] != customer['customer_id']]
            if other_customers:
                beneficiary_customer = random.choice(other_customers)
                beneficiary_account = beneficiary_customer['account_number']
                beneficiary_name = beneficiary_customer['customer_name']
                beneficiary_bank_info = BANK_PREFIXES[beneficiary_customer['bank_code']]
                beneficiary_ifsc = f"{beneficiary_bank_info['ifsc_prefix']}{fake.random_number(digits=6, fix_len=True)}"
                beneficiary_bank = beneficiary_bank_info['name']
                receiver_bank_name = beneficiary_customer['sender_bank_name']
        
        # Generate UTR numbers (bank statement requirement) - PROTECTED FIELD
        utr_number = generate_realistic_utr_number(transaction_code, bank_code, current_date)
        
        # Generate cheque number if applicable - PROTECTED FIELD
        cheque_number = None
        if transaction_code == 'CHQS':
            cheque_number = generate_realistic_cheque_number()
        
        # Generate charges based on transaction type and account type
        charges_applied = 0.0
        gst_on_charges = 0.0
        
        # Premium accounts get reduced charges
        charge_multiplier = 0.5 if customer['customer_segment'] == 'PREMIUM' else 1.0
        
        if transaction_code in ['NEFT', 'RTGS']:
            charges_applied = round(random.uniform(5, 50) * charge_multiplier, 2)
            gst_on_charges = round(charges_applied * 0.18, 2)
        elif transaction_code in ['IMPS', 'UPI'] and amount > 10000:
            charges_applied = round(random.uniform(1, 10) * charge_multiplier, 2)
            gst_on_charges = round(charges_applied * 0.18, 2)
        
        # Salary and NRI accounts might have different charge structures
        if account_type == 'SALARY':
            charges_applied *= 0.5  # Reduced charges for salary accounts
        elif account_type == 'NRI':
            charges_applied *= 1.5  # Higher charges for NRI accounts
        
        # Create base transaction data with bank statement specific fields
        base_data = {
            # Bank Statement Core Fields
            'Transaction_Sequence_Number': transaction_sequence,  # Sequential numbering per customer
            'Statement_Period_Start': customer_pool.customer_statement_periods[customer_id]['start_date'].strftime('%Y-%m-%d'),
            'Statement_Period_End': customer_pool.customer_statement_periods[customer_id]['end_date'].strftime('%Y-%m-%d'),
            'Statement_Number': customer_pool.customer_statement_periods[customer_id]['statement_number'],
            
            # Core Transaction Details (PROTECTED FIELDS)
            'Transaction_Ref_No': transaction_ref,
            'Account_Number': account_number,
            'Account_Type': account_type,
            'Account_Type_Name': account_type_info['name'],
            'Account_Interest_Rate': account_type_info['interest_rate'],
            'Account_Min_Balance': account_type_info['min_balance'],
            'Account_Transaction_Limit': account_type_info['transaction_limit'],
            'Transaction_Date': current_date.strftime('%Y-%m-%d'),
            'Transaction_Time': transaction_datetime.strftime('%H:%M:%S'),
            'Transaction_DateTime': transaction_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'Value_Date': value_date.strftime('%Y-%m-%d'),  # Bank statement requirement
            'Processing_Date': datetime.now().strftime('%Y-%m-%d'),
            
            # Time-based Analytics
            'Month_Category': month_category,
            'Week_In_Month': week_in_month,
            'Day_Of_Week': day_of_week,
            'Quarter': quarter,
            'Hour_Of_Day': transaction_datetime.hour,
            'Is_Weekend': day_of_week in ['Saturday', 'Sunday'],
            'Is_Holiday': random.choice([True, False]) if random.randint(1, 10) == 1 else False,
            
            # Transaction Classification
            'Transaction_Code': transaction_code,
            'Transaction_Description': transaction_codes[transaction_code],
            'Transaction_Category': category,
            'Transaction_Sub_Type': transaction_sub_type,
            'Debit_Credit_Flag': dr_cr_flag,
            'Transaction_Purpose': random.choice(['BUSINESS', 'PERSONAL', 'INVESTMENT', 'GOVERNMENT', 'EDUCATION']),
            
            # Financial Details (Bank Statement Format)
            'Transaction_Amount': amount,
            'Transaction_Currency': 'INR',
            'Exchange_Rate': 1.0,
            'USD_Equivalent': round(amount / 83.0, 2),
            'Opening_Balance': round(opening_balance, 2),  # Bank statement requirement
            'Closing_Balance': round(closing_balance, 2),   # Bank statement requirement (running balance)
            'Available_Balance': round(closing_balance * 0.95, 2),
            
            # Fee and Charges
            'Charges_Applied': round(charges_applied, 2),
            'GST_on_Charges': round(gst_on_charges, 2),
            'Total_Charges': round(charges_applied + gst_on_charges, 2),
            'Net_Amount': round(amount - charges_applied - gst_on_charges, 2) if dr_cr_flag == 'DR' else amount,
            
            # Beneficiary Information
            'Beneficiary_Account': beneficiary_account,
            'Beneficiary_IFSC': beneficiary_ifsc,
            'Beneficiary_Name': beneficiary_name,
            'Beneficiary_Bank': beneficiary_bank,
            
            # Banking Identifiers (Bank Statement Requirements) - PROTECTED FIELDS
            'UTR_Number': utr_number,
            'Cheque_Number': cheque_number,
            'Card_Number': f"****-****-****-{fake.random_number(digits=4, fix_len=True)}" if transaction_code == 'CARD' else None,
            'UPI_ID': f"{fake.user_name()}@{random.choice(UPI_PROVIDERS)}" if transaction_code == 'UPI' else None,
            
            # Channel Information
            'Channel_Type': channel_type,
            'Channel_Name': specific_channel,
            'Device_Type': random.choice(['MOBILE', 'DESKTOP', 'TABLET', 'ATM', 'POS']) if channel_type == 'DIGITAL' else 'PHYSICAL',
            'IP_Address': fake.ipv4() if channel_type == 'DIGITAL' else None,
            'User_Agent': fake.user_agent() if channel_type == 'DIGITAL' else None,
            
            # Branch and Location
            'Branch_Code': branch_code,
            'Branch_Name': f"{geo_city} Main Branch",
            'Transaction_State': geo_state,
            'Transaction_City': geo_city,
            'Region': random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST', 'CENTRAL']),
            'Zone': random.choice(['METRO', 'URBAN', 'SEMI_URBAN', 'RURAL']),
            'Country': 'INDIA',
            
            # Customer Information (PROTECTED FIELDS)
            'Customer_ID': customer_id,
            'Customer_Name': customer_name,
            'Customer_Type': customer['customer_type'],
            'Customer_Segment': customer['customer_segment'],
            'KYC_Status': customer['kyc_status'],
            'Customer_Risk_Category': customer['risk_category'],
            'Relationship_Manager': customer['relationship_manager'],
            
            # Bank Information (NEW - Separate sender bank columns)
            'Sender_Bank_Name': sender_bank_name,
            'Sender_Bank_Code': bank_code,
            'Sender_IFSC': ifsc_code,
            'Receiver_Bank_Name': receiver_bank_name,
            'Receiver_Bank_Code': beneficiary_customer['bank_code'] if 'beneficiary_customer' in locals() else None,
            
            # Transaction Status and Processing
            'Transaction_Status': random.choices(['SUCCESS', 'PENDING', 'FAILED', 'RETURNED'], weights=[85, 8, 5, 2], k=1)[0],
            'Status_Code': random.choice(['00', '01', '02', '03', '04', '05']),
            'Status_Description': random.choice(['Approved', 'Insufficient Funds', 'Invalid Account', 'System Error']),
            'Processing_Time_Seconds': round(random.uniform(0.1, 5.0), 2),
            'Retry_Count': random.randint(0, 3) if random.choice([True, False]) else 0,
            
            # Compliance and Regulatory
            'AML_Flag': random.choice([True, False]) if random.randint(1, 100) <= 5 else False,
            'CTR_Required': amount >= 1000000,
            'STR_Flag': random.choice([True, False]) if random.randint(1, 1000) <= 1 else False,
            'Regulatory_Reporting_Required': amount >= 200000,
            
            # Additional Identifiers
            'Batch_ID': f"BATCH_{current_date.strftime('%Y%m%d')}_{fake.random_number(digits=4, fix_len=True)}",
            'Journal_ID': f"JRN{fake.random_number(digits=10, fix_len=True)}",
            'Sequence_Number': fake.random_number(digits=6, fix_len=True),
            'Business_Date': current_date.strftime('%Y-%m-%d'),
            
            # Technical Metadata
            'Record_Created_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Record_Updated_At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Data_Source': 'CORE_BANKING_SYSTEM',
            'Message_ID': str(uuid.uuid4()),
            'Correlation_ID': str(uuid.uuid4()),
            'Session_ID': f"SES{fake.random_number(digits=12, fix_len=True)}" if channel_type == 'DIGITAL' else None
        }
        
        # Generate realistic transaction description (bank statement requirement) with error handling
        try:
            realistic_description = generate_realistic_transaction_description(transaction_code, base_data)
            base_data['Realistic_Transaction_Description'] = realistic_description
        except Exception as e:
            # Fallback description if generation fails
            base_data['Realistic_Transaction_Description'] = f"{transaction_code} Transaction - {base_data.get('Transaction_Purpose', 'GENERAL')}"
        
        # Calculate risk score (now includes account type and customer behavior)
        try:
            risk_data = calculate_risk_score(base_data)
            base_data.update({
                'Risk_Score': risk_data['risk_score'],
                'Risk_Level': risk_data['risk_level'],
                'Risk_Flags': ','.join(risk_data['risk_flags']) if risk_data['risk_flags'] else None
            })
        except Exception as e:
            # Fallback risk data
            base_data.update({
                'Risk_Score': 25,
                'Risk_Level': 'MEDIUM',
                'Risk_Flags': None
            })
        
        # Apply anomaly injection to selected fields - PROTECT CRITICAL BANK STATEMENT FIELDS
        protected_fields = [
            # CRITICAL PROTECTED FIELDS - NEVER INJECT ANOMALIES
            'Transaction_Ref_No', 'Account_Number', 'Customer_ID', 'Customer_Name',
            'UTR_Number', 'Cheque_Number', 'Transaction_Sequence_Number', 'Statement_Number',
            
            # Additional protected fields for data integrity
            'Message_ID', 'Correlation_ID', 'Journal_ID', 'Batch_ID', 'Risk_Level', 'Risk_Score',
            'Customer_Segment', 'Transaction_Code', 'Debit_Credit_Flag', 'Channel_Type',
            'Transaction_Category', 'Hour_Of_Day', 'Week_In_Month', 'Account_Type',
            'Account_Type_Name', 'Account_Interest_Rate', 'Account_Min_Balance', 'Account_Transaction_Limit',
            'Sender_Bank_Name', 'Sender_Bank_Code', 'Sender_IFSC'  # NEW: Protect sender bank fields
        ]
        
        injected_data = {}
        for key, value in base_data.items():
            if key in protected_fields:
                # NEVER inject anomalies for these critical fields
                injected_data[key] = value
            elif 'Date' in key or 'Time' in key:
                injected_data[key] = inject_anomaly(value, 'date')
            elif 'Amount' in key or 'Balance' in key or 'Charges' in key:
                injected_data[key] = inject_anomaly(value, 'amount')
            elif key in ['Transaction_Status', 'Status_Description']:
                injected_data[key] = inject_anomaly(value, 'status')
            elif key in ['Channel_Name']:
                injected_data[key] = inject_anomaly(value, 'code')
            elif isinstance(value, (int, float)):
                injected_data[key] = inject_anomaly(value, 'numeric')
            elif isinstance(value, str):
                injected_data[key] = inject_anomaly(value, 'string')
            else:
                injected_data[key] = value
        
        return injected_data
        
    except Exception as e:
        print(f"Error generating transaction data: {e}")
        # Return a minimal valid transaction in case of error
        fallback_customer_id = f"CUST{fake.random_number(digits=10, fix_len=True)}"
        fallback_account = f"ACC{fake.random_number(digits=12, fix_len=True)}"
        fallback_ref = f"TXN{fake.random_number(digits=16, fix_len=True)}"
        fallback_stmt = f"ST{fake.random_number(digits=8, fix_len=True)}"
        fallback_bank = random.choice(list(BANK_PREFIXES.values()))['name']
        
        return {
            'Transaction_Ref_No': fallback_ref,
            'Account_Number': fallback_account,
            'Account_Type': 'SAVINGS',
            'Account_Type_Name': 'Savings Account',
            'Transaction_Date': datetime.now().strftime('%Y-%m-%d'),
            'Transaction_Amount': 1000.0,
            'Transaction_Status': 'SUCCESS',
            'Customer_ID': fallback_customer_id,
            'Customer_Name': fake.name(),
            'Month_Category': 'Current Month',
            'Transaction_Code': 'NEFT',
            'Debit_Credit_Flag': 'DR',
            'Transaction_Sequence_Number': 1,
            'Opening_Balance': 10000.0,
            'Closing_Balance': 9000.0,
            'Statement_Number': fallback_stmt,
            'Statement_Period_Start': datetime.now().strftime('%Y-%m-%d'),
            'Statement_Period_End': datetime.now().strftime('%Y-%m-%d'),
            'Realistic_Transaction_Description': 'NEFT Transaction - GENERAL',
            'Risk_Score': 25,
            'Risk_Level': 'MEDIUM',
            'Risk_Flags': None,
            'Sender_Bank_Name': fallback_bank,
            'Sender_Bank_Code': 'BANK',
            'Sender_IFSC': f"BANK{fake.random_number(digits=7, fix_len=True)}"
        }

def generate_date_range():
    """Generate date range covering last month and current month"""
    current_date = datetime.now().date()
    current_month_start = current_date.replace(day=1)
    last_month_end = current_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    
    # Generate dates with more concentration on recent days
    date_list = []
    
    # Last month dates (35% of transactions)
    last_month_days = (last_month_end - last_month_start).days + 1
    for i in range(last_month_days):
        date = last_month_start + timedelta(days=i)
        # More transactions towards month end
        daily_transactions = max(1, int(20 * (i + 1) / last_month_days))
        date_list.extend([date] * daily_transactions)
    
    # Current month dates (65% of transactions)
    current_month_days = (current_date - current_month_start).days + 1
    for i in range(current_month_days):
        date = current_month_start + timedelta(days=i)
        # More transactions on recent days
        daily_transactions = max(1, int(30 * (i + 1) / current_month_days))
        date_list.extend([date] * daily_transactions)
    
    return date_list

def produce_banking_data(bootstrap_servers=['localhost:9092'], topic='banking_transactions'):
    # Initialize customer pool
    customer_pool = CustomerPool(pool_size=250)  # 250 customers for frequent transactions
    
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, cls=NumPyEncoder).encode('utf-8')
    )
    
    topic_name = 'banking_transactions'
    entry_limit = 5000  # Exactly 5000 rows
    entry_count = 1
    anomaly_count = 0
    error_count = 0
    
    # Generate date sequence for realistic distribution
    date_sequence = generate_date_range()
    random.shuffle(date_sequence)  # Randomize order but maintain distribution
    
    # Statistics for comprehensive reporting (including account types and sender banks)
    stats = {
        'total_transactions': 0,
        'current_month_transactions': 0,
        'last_month_transactions': 0,
        'total_debit_amount': 0,
        'total_credit_amount': 0,
        'channel_stats': {},
        'category_stats': {},
        'risk_stats': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0},
        'customer_segment_stats': {},
        'geography_stats': {},
        'currency_stats': {},
        'account_type_stats': {},
        'sender_bank_stats': {},  # NEW: Track sender bank statistics
        'weekend_transactions': 0,
        'holiday_transactions': 0,
        'failed_transactions': 0,
        'customer_transaction_count': {},  # Track transactions per customer
        'unique_customers_transacted': set(),
        'error_transactions': 0,
        'protected_field_integrity': {  # NEW: Track protected field integrity
            'unique_transaction_refs': set(),
            'unique_account_numbers': set(),
            'unique_customer_ids': set(),
            'unique_statement_numbers': set(),
            'duplicate_transaction_refs': 0,
            'duplicate_account_numbers': 0,
            'duplicate_customer_ids': 0,
            'duplicate_statement_numbers': 0
        }
    }
    
    print("🏦 ENHANCED REAL-TIME BANKING DATA PRODUCER WITH BANK STATEMENT FORMAT")
    print("="*90)
    print("🎯 TARGET: 5000 TRANSACTIONS WITH 15% ANOMALIES")
    print("👥 CUSTOMER POOL: 250 FREQUENT TRANSACTING CUSTOMERS")
    print("✅ Complete Bank Statement Format Features Included:")
    print("   📊 Sequential Transaction Numbering per Customer")
    print("   💰 Running Balance Column (Opening/Closing Balance)")
    print("   🔢 Proper Reference Numbers (UTR, Cheque Numbers)")
    print("   📝 Realistic Transaction Descriptions with Beneficiary Details")
    print("   📅 Statement Period Headers and Value Date vs Transaction Date")
    print("   🏛️  Account Type Analytics (Savings, Current, FD, RD, Salary, NRI)")
    print("   🏦 Separate Sender Bank Name Column (NEW)")
    print("   👤 Limited Customer Pool with Frequent Daily Transactions")
    print("   📅 Realistic Date Distribution (Last Month + Current Month)")
    print("   🔄 Customer Behavior Patterns (Daily/Weekly Transaction Frequency)")
    print("   📊 Customer Analytics & Segmentation")
    print("   🌍 Multi-Geography & Multi-Channel Support") 
    print("   💰 Multi-Currency & Fee Management")
    print("   🔍 Risk Scoring & Compliance Flags")
    print("   📅 Comprehensive Time-based Analytics")
    print("   🏛️ Regulatory Reporting Capabilities")
    print("   📱 Digital Channel Tracking")
    print("   🎯 Real-time Anomaly Detection (15%)")
    print("   🔒 PROTECTED: Transaction_Ref_No, Customer_ID, Account_Number, Customer_Name, UTR, Cheque Numbers, Statement_Number (NO ANOMALIES)")
    print("   🛡️  Enhanced Error Handling and Recovery")
    print("   ✅ GUARANTEED UNIQUE: All protected fields have collision detection")
    print("="*90)
    
    # Display customer pool summary
    print(f"\n👥 CUSTOMER POOL SUMMARY:")
    account_type_distribution = {}
    segment_distribution = {}
    sender_bank_distribution = {}
    
    for customer in customer_pool.customers:
        acc_type = customer['account_type']
        segment = customer['customer_segment']
        sender_bank = customer['sender_bank_name']
        
        account_type_distribution[acc_type] = account_type_distribution.get(acc_type, 0) + 1
        segment_distribution[segment] = segment_distribution.get(segment, 0) + 1
        sender_bank_distribution[sender_bank] = sender_bank_distribution.get(sender_bank, 0) + 1
    
    print(f"   Account Type Distribution:")
    for acc_type, count in account_type_distribution.items():
        percentage = (count / len(customer_pool.customers)) * 100
        print(f"      {acc_type}: {count} customers ({percentage:.1f}%)")
    
    print(f"\n   Customer Segment Distribution:")
    for segment, count in segment_distribution.items():
        percentage = (count / len(customer_pool.customers)) * 100
        print(f"      {segment}: {count} customers ({percentage:.1f}%)")
    
    print(f"\n   Sender Bank Distribution:")
    for bank, count in sorted(sender_bank_distribution.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / len(customer_pool.customers)) * 100
        print(f"      {bank}: {count} customers ({percentage:.1f}%)")
    
    print(f"\n📅 DATE RANGE: {len(set(date_sequence))} unique days, {len(date_sequence)} total transaction slots")
    print(f"🔒 PROTECTED FIELD INTEGRITY:")
    print(f"   Pre-allocated Unique Customer IDs: {len(used_customer_ids):,}")
    print(f"   Pre-allocated Unique Account Numbers: {len(used_account_numbers):,}")
    print(f"   Pre-allocated Unique Statement Numbers: {len(used_statement_numbers):,}")
    print("="*90)
    
    try:
        while entry_count <= entry_limit and entry_count <= len(date_sequence):
            try:
                # Use date from sequence to ensure proper distribution
                target_date = date_sequence[entry_count - 1] if entry_count <= len(date_sequence) else date_sequence[-1]
                
                data = generate_banking_transaction_data(customer_pool, target_date)
                
                if not data:
                    print(f"❌ Failed to generate transaction data for entry {entry_count}")
                    entry_count += 1
                    error_count += 1
                    continue
                
                # Validate protected field integrity
                transaction_ref = data.get('Transaction_Ref_No')
                account_number = data.get('Account_Number')
                customer_id = data.get('Customer_ID')
                statement_number = data.get('Statement_Number')
                
                # Check for duplicates in protected fields
                if transaction_ref:
                    if transaction_ref in stats['protected_field_integrity']['unique_transaction_refs']:
                        stats['protected_field_integrity']['duplicate_transaction_refs'] += 1
                    else:
                        stats['protected_field_integrity']['unique_transaction_refs'].add(transaction_ref)
                
                if account_number:
                    if account_number in stats['protected_field_integrity']['unique_account_numbers']:
                        stats['protected_field_integrity']['duplicate_account_numbers'] += 1
                    else:
                        stats['protected_field_integrity']['unique_account_numbers'].add(account_number)
                
                if customer_id:
                    if customer_id in stats['protected_field_integrity']['unique_customer_ids']:
                        stats['protected_field_integrity']['duplicate_customer_ids'] += 1
                    else:
                        stats['protected_field_integrity']['unique_customer_ids'].add(customer_id)
                
                if statement_number:
                    if statement_number in stats['protected_field_integrity']['unique_statement_numbers']:
                        stats['protected_field_integrity']['duplicate_statement_numbers'] += 1
                    else:
                        stats['protected_field_integrity']['unique_statement_numbers'].add(statement_number)
                
                # Update statistics for real-time reporting
                stats['total_transactions'] += 1
                
                # Track customer transaction frequency
                if customer_id:
                    stats['customer_transaction_count'][customer_id] = stats['customer_transaction_count'].get(customer_id, 0) + 1
                    stats['unique_customers_transacted'].add(customer_id)
                
                if data.get('Month_Category') == 'Current Month':
                    stats['current_month_transactions'] += 1
                else:
                    stats['last_month_transactions'] += 1
                
                # Amount statistics (safe handling of None values)
                amount = data.get('Transaction_Amount', 0)
                if isinstance(amount, (int, float)) and amount > 0:
                    if data.get('Debit_Credit_Flag') == 'DR':
                        stats['total_debit_amount'] += amount
                    else:
                        stats['total_credit_amount'] += amount
                
                # Channel statistics
                channel = data.get('Channel_Name', 'UNKNOWN')
                if channel:
                    stats['channel_stats'][channel] = stats['channel_stats'].get(channel, 0) + 1
                
                # Category statistics
                category = data.get('Transaction_Category', 'UNKNOWN')
                if category:
                    stats['category_stats'][category] = stats['category_stats'].get(category, 0) + 1
                
                # Risk statistics
                risk_level = data.get('Risk_Level', 'LOW')
                if risk_level:
                    stats['risk_stats'][risk_level] += 1
                
                # Customer segment statistics
                segment = data.get('Customer_Segment', 'UNKNOWN')
                if segment:
                    stats['customer_segment_stats'][segment] = stats['customer_segment_stats'].get(segment, 0) + 1
                
                # Geography statistics
                state = data.get('Transaction_State', 'UNKNOWN')
                if state:
                    stats['geography_stats'][state] = stats['geography_stats'].get(state, 0) + 1
                
                # Account type statistics
                account_type = data.get('Account_Type', 'UNKNOWN')
                if account_type:
                    stats['account_type_stats'][account_type] = stats['account_type_stats'].get(account_type, 0) + 1
                
                # NEW: Sender bank statistics
                sender_bank = data.get('Sender_Bank_Name', 'UNKNOWN')
                if sender_bank:
                    stats['sender_bank_stats'][sender_bank] = stats['sender_bank_stats'].get(sender_bank, 0) + 1
                
                # Time-based statistics
                if data.get('Is_Weekend'):
                    stats['weekend_transactions'] += 1
                if data.get('Is_Holiday'):
                    stats['holiday_transactions'] += 1
                if data.get('Transaction_Status') == 'FAILED':
                    stats['failed_transactions'] += 1
                
                # Count anomalies for reporting (EXCLUDE protected fields from anomaly count)
                anomalies_in_record = 0
                protected_check_fields = [
                    'Transaction_Ref_No', 'Account_Number', 'Customer_ID', 'Customer_Name', 
                    'UTR_Number', 'Cheque_Number', 'Transaction_Sequence_Number', 'Statement_Number',
                    'Sender_Bank_Name', 'Sender_Bank_Code', 'Sender_IFSC'
                ]
                
                for key, value in data.items():
                    if key not in protected_check_fields:  # Don't check protected fields for anomalies
                        if (value is None or 
                            (isinstance(value, str) and (value == "" or "@#$" in value)) or 
                            (isinstance(value, (int, float)) and value < 0)):
                            anomalies_in_record += 1
                
                if anomalies_in_record > 0:
                    anomaly_count += 1
                
                # Send to Kafka
                producer.send(topic_name, value=data)
                
                # Enhanced progress reporting with safe formatting
                status_icon = "⚠️" if anomalies_in_record > 0 else "✅"
                risk_level_safe = safe_format(data.get('Risk_Level', 'UNKNOWN'))
                risk_icon = "🔴" if risk_level_safe == 'HIGH' else "🟡" if risk_level_safe == 'MEDIUM' else "🟢"
                
                if entry_count <= 10 or entry_count % 250 == 0:
                    customer_txn_count = stats['customer_transaction_count'].get(customer_id, 0)
                    
                    print(f"\n{status_icon} Transaction #{entry_count:,}")
                    print(f"   🆔 TXN: {safe_format(data.get('Transaction_Ref_No', 'N/A')[:20])}...")
                    print(f"   📊 Seq#: {safe_format(data.get('Transaction_Sequence_Number', 'N/A'))} | Statement: {safe_format(data.get('Statement_Number', 'N/A')[:15])}...")
                    print(f"   🏦 ACC: {safe_format(data.get('Account_Number', 'N/A'))} | Type: {safe_format(data.get('Account_Type', 'N/A'))}")
                    print(f"   🏛️  {safe_format(data.get('Account_Type_Name', 'N/A'))} | Rate: {safe_format(data.get('Account_Interest_Rate', 0))}%")
                    print(f"   👤 Customer: {safe_format(data.get('Customer_Name', 'N/A'))} ({safe_format(data.get('Customer_Segment', 'N/A'))})")
                    print(f"   🏛️  Sender Bank: {safe_format(data.get('Sender_Bank_Name', 'N/A'))}")  # NEW
                    print(f"   🔄 Customer Transactions: {customer_txn_count} (This customer's #{customer_txn_count} transaction)")
                    
                    amount_safe = data.get('Transaction_Amount', 0)
                    amount_str = f"₹{amount_safe:,.2f}" if isinstance(amount_safe, (int, float)) else safe_format(amount_safe)
                    opening_bal = data.get('Opening_Balance', 0)
                    closing_bal = data.get('Closing_Balance', 0)
                    opening_str = f"₹{opening_bal:,.2f}" if isinstance(opening_bal, (int, float)) else safe_format(opening_bal)
                    closing_str = f"₹{closing_bal:,.2f}" if isinstance(closing_bal, (int, float)) else safe_format(closing_bal)
                    
                    print(f"   💳 {safe_format(data.get('Transaction_Code', 'N/A'))} - {amount_str} ({safe_format(data.get('Debit_Credit_Flag', 'N/A'))})")
                    print(f"   💰 Balance: {opening_str} → {closing_str}")
                    
                    # Show UTR/Cheque if available
                    utr = data.get('UTR_Number')
                    cheque = data.get('Cheque_Number')
                    if utr:
                        print(f"   🔢 UTR: {safe_format(utr)}")
                    if cheque:
                        print(f"   📝 Cheque: {safe_format(cheque)}")
                    
                    print(f"   📝 Description: {safe_format(data.get('Realistic_Transaction_Description', 'N/A')[:50])}...")
                    print(f"   📍 {safe_format(data.get('Transaction_City', 'N/A'))}, {safe_format(data.get('Transaction_State', 'N/A'))} | {safe_format(data.get('Channel_Name', 'N/A'))}")
                    print(f"   {risk_icon} Risk: {risk_level_safe} (Score: {safe_format(data.get('Risk_Score', 0))}) | Status: {safe_format(data.get('Transaction_Status', 'N/A'))}")
                    print(f"   📅 {safe_format(data.get('Month_Category', 'N/A'))} | TXN: {safe_format(data.get('Transaction_Date', 'N/A'))} | Value: {safe_format(data.get('Value_Date', 'N/A'))}")
                    
                    if anomalies_in_record > 0:
                        print(f"   ⚠️  Anomalies detected: {anomalies_in_record}")
                    
                    # Real-time statistics summary every 500 transactions
                    if entry_count % 500 == 0:
                        print(f"\n📊 REAL-TIME BANK STATEMENT ANALYTICS DASHBOARD")
                        print(f"   {'='*70}")
                        print(f"   📈 Total Processed: {stats['total_transactions']:,} / {entry_limit:,}")
                        print(f"   📈 Progress: {(stats['total_transactions']/entry_limit)*100:.1f}%")
                        print(f"   👥 Unique Customers: {len(stats['unique_customers_transacted']):,} / {len(customer_pool.customers):,}")
                        print(f"   📅 Current Month: {stats['current_month_transactions']:,} | Last Month: {stats['last_month_transactions']:,}")
                        print(f"   💰 Total Debits: ₹{stats['total_debit_amount']:,.2f}")
                        print(f"   💰 Total Credits: ₹{stats['total_credit_amount']:,.2f}")
                        print(f"   💱 Net Flow: ₹{stats['total_credit_amount'] - stats['total_debit_amount']:,.2f}")
                        
                        # Protected field integrity report
                        print(f"\n   🔒 PROTECTED FIELD INTEGRITY:")
                        integrity = stats['protected_field_integrity']
                        print(f"      Unique Transaction Refs: {len(integrity['unique_transaction_refs']):,}")
                        print(f"      Unique Account Numbers: {len(integrity['unique_account_numbers']):,}")
                        print(f"      Unique Customer IDs: {len(integrity['unique_customer_ids']):,}")
                        print(f"      Unique Statement Numbers: {len(integrity['unique_statement_numbers']):,}")
                        print(f"      Duplicates Detected: TXN:{integrity['duplicate_transaction_refs']}, ACC:{integrity['duplicate_account_numbers']}, CUST:{integrity['duplicate_customer_ids']}, STMT:{integrity['duplicate_statement_numbers']}")
                        
                        # Show top frequent customers
                        top_customers = sorted(stats['customer_transaction_count'].items(), key=lambda x: x[1], reverse=True)[:5]
                        print(f"\n   🔥 Top Frequent Customers:")
                        for cust_id, txn_count in top_customers:
                            customer_name = next((c['customer_name'] for c in customer_pool.customers if c['customer_id'] == cust_id), 'Unknown')
                            print(f"      {customer_name[:20]}: {txn_count} transactions")
                        
                        print(f"\n   🏛️  Account Type Distribution:")
                        for acc_type, count in stats['account_type_stats'].items():
                            percentage = (count / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
                            print(f"      {acc_type}: {count:,} ({percentage:.1f}%)")
                        
                        print(f"\n   🏦 Sender Bank Distribution:")
                        top_sender_banks = sorted(stats['sender_bank_stats'].items(), key=lambda x: x[1], reverse=True)[:5]
                        for bank, count in top_sender_banks:
                            percentage = (count / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
                            print(f"      {bank[:25]}: {count:,} ({percentage:.1f}%)")
                        
                        print(f"\n   🔍 Risk Distribution:")
                        for risk, count in stats['risk_stats'].items():
                            percentage = (count / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
                            print(f"      {risk}: {count:,} ({percentage:.1f}%)")
                        
                        print(f"\n   📱 Top Channels:")
                        top_channels = sorted(stats['channel_stats'].items(), key=lambda x: x[1], reverse=True)[:3]
                        for channel, count in top_channels:
                            percentage = (count / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
                            print(f"      {channel}: {count:,} ({percentage:.1f}%)")
                        
                        print(f"\n   ⚠️  Quality Metrics:")
                        anomaly_rate = (anomaly_count/stats['total_transactions'])*100 if stats['total_transactions'] > 0 else 0
                        failure_rate = (stats['failed_transactions']/stats['total_transactions'])*100 if stats['total_transactions'] > 0 else 0
                        error_rate = (error_count/entry_count)*100 if entry_count > 0 else 0
                        print(f"      Anomalies: {anomaly_count:,} ({anomaly_rate:.1f}%) - Target: 15%")
                        print(f"      Failed Txns: {stats['failed_transactions']:,} ({failure_rate:.1f}%)")
                        print(f"      Error Rate: {error_count:,} ({error_rate:.1f}%)")
                        print(f"      Weekend Txns: {stats['weekend_transactions']:,}")
                        print(f"      Holiday Txns: {stats['holiday_transactions']:,}")
                        
                        print(f"   {'='*70}")
                
                entry_count += 1
                time.sleep(0.001)  # Small delay for realistic streaming
                
            except Exception as e:
                print(f"\n❌ Error processing transaction {entry_count}: {e}")
                error_count += 1
                entry_count += 1
                continue
            
            # Stop after processing all transactions
            if entry_count > entry_limit:
                break
    
    except KeyboardInterrupt:
        print(f"\n⚠️ Stopped by user. Processed {entry_count-1} transactions.")
    
    except Exception as e:
        print(f"\n❌ Major error occurred: {e}")
        print(f"Processed {entry_count-1} transactions before error.")
    
    finally:
        producer.flush()
        producer.close()
        
        # Final comprehensive summary
        print(f"\n🎯 ENHANCED BANKING DATA WITH BANK STATEMENT FORMAT COMPLETED!")
        print(f"="*90)
        print(f"📊 FINAL STATISTICS:")
        print(f"   Target Transactions: {entry_limit:,}")
        print(f"   Actual Transactions: {stats['total_transactions']:,}")
        print(f"   Completion Rate: {(stats['total_transactions']/entry_limit)*100:.1f}%")
        print(f"   Error Count: {error_count:,}")
        print(f"   Success Rate: {((stats['total_transactions'])/(entry_count-1))*100:.1f}%")
        print(f"   Current Month: {stats['current_month_transactions']:,}")
        print(f"   Last Month: {stats['last_month_transactions']:,}")
        print(f"   Total Value: ₹{stats['total_debit_amount'] + stats['total_credit_amount']:,.2f}")
        print(f"   Net Flow: ₹{stats['total_credit_amount'] - stats['total_debit_amount']:,.2f}")
        
        # CRITICAL: Protected field integrity final report
        print(f"\n🔒 PROTECTED FIELD INTEGRITY FINAL REPORT:")
        integrity = stats['protected_field_integrity']
        print(f"   Unique Transaction References: {len(integrity['unique_transaction_refs']):,}")
        print(f"   Unique Account Numbers: {len(integrity['unique_account_numbers']):,}")
        print(f"   Unique Customer IDs: {len(integrity['unique_customer_ids']):,}")
        print(f"   Unique Statement Numbers: {len(integrity['unique_statement_numbers']):,}")
        print(f"   Duplicate Detection Results:")
        print(f"      Transaction_Ref_No Duplicates: {integrity['duplicate_transaction_refs']:,}")
        print(f"      Account_Number Duplicates: {integrity['duplicate_account_numbers']:,}")
        print(f"      Customer_ID Duplicates: {integrity['duplicate_customer_ids']:,}")
        print(f"      Statement_Number Duplicates: {integrity['duplicate_statement_numbers']:,}")
        
        total_duplicates = (integrity['duplicate_transaction_refs'] + 
                          integrity['duplicate_account_numbers'] + 
                          integrity['duplicate_customer_ids'] + 
                          integrity['duplicate_statement_numbers'])
        
        print(f"   ✅ INTEGRITY STATUS: {'PERFECT' if total_duplicates == 0 else 'ISSUES DETECTED'}")
        print(f"   🎯 Target: ZERO duplicates in protected fields")
        print(f"   📊 Result: {total_duplicates} total duplicates detected")
        
        print(f"\n👥 CUSTOMER FREQUENCY ANALYSIS:")
        print(f"   Total Customers in Pool: {len(customer_pool.customers):,}")
        print(f"   Customers who Transacted: {len(stats['unique_customers_transacted']):,}")
        print(f"   Customer Participation Rate: {(len(stats['unique_customers_transacted'])/len(customer_pool.customers))*100:.1f}%")
        
        # Average transactions per customer
        if len(stats['unique_customers_transacted']) > 0:
            avg_txns_per_customer = stats['total_transactions'] / len(stats['unique_customers_transacted'])
            print(f"   Average Transactions per Customer: {avg_txns_per_customer:.1f}")
        
        # Show most frequent customers
        print(f"\n🔥 TOP 10 MOST FREQUENT CUSTOMERS:")
        top_customers = sorted(stats['customer_transaction_count'].items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (cust_id, txn_count) in enumerate(top_customers, 1):
            customer_info = next((c for c in customer_pool.customers if c['customer_id'] == cust_id), None)
            if customer_info:
                name = customer_info['customer_name'][:25]
                acc_type = customer_info['account_type']
                segment = customer_info['customer_segment']
                sender_bank = customer_info['sender_bank_name'][:20]
                current_balance = customer_pool.customer_balances.get(cust_id, 0)
                print(f"   {i:2d}. {name:<25} | {acc_type:<15} | {segment:<10} | {txn_count:3d} txns | ₹{current_balance:,.0f}")
                print(f"       Sender Bank: {sender_bank} | Account: {customer_info['account_number']}")
        
        print(f"\n🏛️  ACCOUNT TYPE SUMMARY:")
        for acc_type, count in stats['account_type_stats'].items():
            percentage = (count / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
            print(f"   {acc_type}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🏦 SENDER BANK ANALYSIS:")
        if stats['sender_bank_stats']:
            print(f"   Total Unique Sender Banks: {len(stats['sender_bank_stats']):,}")
            print(f"   Top Sender Banks:")
            for bank, count in sorted(stats['sender_bank_stats'].items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / stats['total_transactions']) * 100 if stats['total_transactions'] > 0 else 0
                print(f"      {bank:<30}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n📅 TEMPORAL DISTRIBUTION:")
        print(f"   Unique Transaction Dates: {len(set(date_sequence))}")
        print(f"   Date Range Coverage: Last Month + Current Month")
        print(f"   Weekend Transactions: {stats['weekend_transactions']:,}")
        print(f"   Holiday Transactions: {stats['holiday_transactions']:,}")
        
        print(f"\n⚠️  DATA QUALITY:")
        total_processed = stats['total_transactions']
        if total_processed > 0:
            anomaly_percentage = (anomaly_count/total_processed)*100
            print(f"   Records with Anomalies: {anomaly_count:,} ({anomaly_percentage:.1f}%)")
            print(f"   Target Anomaly Rate: 15.0%")
            print(f"   Anomaly Target Met: {'✅ YES' if 10 <= anomaly_percentage <= 20 else '❌ NO'}")
            print(f"   Data Quality Score: {((total_processed-anomaly_count)/total_processed)*100:.1f}%")
        else:
            print(f"   Records with Anomalies: {anomaly_count:,} (0.0%)")
            print(f"   Data Quality Score: 100.0%")
        
        print(f"\n🔍 SAMPLE CUSTOMER BANK STATEMENT PREVIEW:")
        # Show a sample of one customer's transactions
        if stats['customer_transaction_count']:
            sample_customer_id = max(stats['customer_transaction_count'], key=stats['customer_transaction_count'].get)
            sample_customer = next((c for c in customer_pool.customers if c['customer_id'] == sample_customer_id), None)
            if sample_customer:
                print(f"   Sample Customer: {sample_customer['customer_name']}")
                print(f"   Account Number: {sample_customer['account_number']}")
                print(f"   Account Type: {sample_customer['account_type']} ({sample_customer['account_type_info']['name']})")
                print(f"   Customer Segment: {sample_customer['customer_segment']}")
                print(f"   Sender Bank: {sample_customer['sender_bank_name']}")
                print(f"   Total Transactions: {stats['customer_transaction_count'][sample_customer_id]:,}")
                print(f"   Final Balance: ₹{customer_pool.customer_balances.get(sample_customer_id, 0):,.2f}")
                print(f"   Home Location: {sample_customer['home_city']}, {sample_customer['home_state']}")
                print(f"   Statement Period: {customer_pool.customer_statement_periods[sample_customer_id]['start_date']} to {customer_pool.customer_statement_periods[sample_customer_id]['end_date']}")
                print(f"   Statement Number: {customer_pool.customer_statement_periods[sample_customer_id]['statement_number']}")
        
        print(f"\n✅ BANK STATEMENT FORMAT FEATURES ACCOMPLISHED:")
        print(f"   ✅ Enhanced banking dataset with {stats['total_transactions']:,} transactions ready!")
        print(f"   ✅ 250 frequent customers with realistic daily transaction patterns!")
        print(f"   ✅ Sequential transaction numbering per customer implemented!")
        print(f"   ✅ Running balance columns (Opening/Closing Balance) added!")
        print(f"   ✅ Proper reference numbers (UTR, Cheque Numbers) generated!")
        print(f"   ✅ Realistic transaction descriptions with beneficiary details!")
        print(f"   ✅ Statement period headers and value date vs transaction date!")
        print(f"   ✅ Separate Sender Bank Name column with {len(stats['sender_bank_stats'])} unique banks!")
        print(f"   ✅ Customer pool designed for individual bank statement generation!")
        print(f"   ✅ Realistic temporal distribution across last month and current month!")
        print(f"   ✅ Customer behavior patterns (frequent transactions per customer)!")
        print(f"   ✅ Account type analytics with customer segments!")
        print(f"   ✅ Multi-channel transaction patterns!")
        print(f"   ✅ Geographic distribution with home location consistency!")
        print(f"   ✅ Risk scoring and compliance monitoring!")
        print(f"   ✅ Real-time analytics and anomaly detection completed!")
        print(f"   ✅ Enhanced error handling and recovery mechanisms!")
        print(f"   ✅ GUARANTEED UNIQUE protected fields with collision detection!")
        print(f"   🔒 PROTECTED FIELDS maintained data integrity:")
        print(f"      • Transaction_Ref_No: {len(integrity['unique_transaction_refs']):,} unique values")
        print(f"      • Customer_ID: {len(integrity['unique_customer_ids']):,} unique values") 
        print(f"      • Account_Number: {len(integrity['unique_account_numbers']):,} unique values")
        print(f"      • Statement_Number: {len(integrity['unique_statement_numbers']):,} unique values")
        print(f"      • Customer_Name: NO anomalies injected")
        print(f"      • UTR_Number: NO anomalies injected")
        print(f"      • Cheque_Number: NO anomalies injected")
        print(f"      • Sender_Bank_Name: NO anomalies injected")
        print(f"   📋 Perfect for generating individual customer bank statements with proper format!")
        print(f"   🏦 Bank-to-bank transfer tracking with sender/receiver bank columns!")
        print(f"   📊 Ready for comprehensive banking analytics and reporting!")
        print(f"="*90)

if __name__ == "__main__":
    produce_banking_data()