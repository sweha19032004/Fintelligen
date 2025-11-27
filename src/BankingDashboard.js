"use client";

import { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  RadialBarChart,
  RadialBar,
} from "recharts";
// Custom icon components to replace lucide-react
const Icon = ({ children, className = "" }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    {children}
  </svg>
);

const TrendingUp = ({ className }) => (
  <Icon className={className}>
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
    <polyline points="17 6 23 6 23 12"></polyline>
  </Icon>
);

const TrendingDown = ({ className }) => (
  <Icon className={className}>
    <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline>
    <polyline points="17 18 23 18 23 12"></polyline>
  </Icon>
);

const DollarSign = ({ className }) => (
  <Icon className={className}>
    <line x1="12" y1="1" x2="12" y2="23"></line>
    <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
  </Icon>
);

const Activity = ({ className }) => (
  <Icon className={className}>
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
  </Icon>
);

const CreditCard = ({ className }) => (
  <Icon className={className}>
    <rect x="1" y="4" width="22" height="16" rx="2" ry="2"></rect>
    <line x1="1" y1="10" x2="23" y2="10"></line>
  </Icon>
);

const AlertCircle = ({ className }) => (
  <Icon className={className}>
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </Icon>
);

const CheckCircle = ({ className }) => (
  <Icon className={className}>
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
    <polyline points="22 4 12 14.01 9 11.01"></polyline>
  </Icon>
);

const RefreshCw = ({ className }) => (
  <Icon className={className}>
    <polyline points="23 4 23 10 17 10"></polyline>
    <polyline points="1 20 1 14 7 14"></polyline>
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
  </Icon>
);

const Brain = ({ className }) => (
  <Icon className={className}>
    <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44A2.5 2.5 0 0 1 2 17.5v-15a2.5 2.5 0 0 1 5-.44A2.5 2.5 0 0 1 9.5 2Z"></path>
    <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44A2.5 2.5 0 0 0 22 17.5v-15a2.5 2.5 0 0 0-5-.44A2.5 2.5 0 0 0 14.5 2Z"></path>
  </Icon>
);

const Calendar = ({ className }) => (
  <Icon className={className}>
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
    <line x1="16" y1="2" x2="16" y2="6"></line>
    <line x1="8" y1="2" x2="8" y2="6"></line>
    <line x1="3" y1="10" x2="21" y2="10"></line>
  </Icon>
);

const FileText = ({ className }) => (
  <Icon className={className}>
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
    <line x1="16" y1="13" x2="8" y2="13"></line>
    <line x1="16" y1="17" x2="8" y2="17"></line>
    <polyline points="10 9 9 9 8 9"></polyline>
  </Icon>
);

const BarChart3 = ({ className }) => (
  <Icon className={className}>
    <path d="M3 3v18h18"></path>
    <path d="M8 17V9"></path>
    <path d="M12 17v-5"></path>
    <path d="M16 17v-2"></path>
  </Icon>
);

const Loader2 = ({ className }) => (
  <Icon className={className}>
    <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
  </Icon>
);

const ArrowUpRight = ({ className }) => (
  <Icon className={className}>
    <line x1="7" y1="17" x2="17" y2="7"></line>
    <polyline points="7 7 17 7 17 17"></polyline>
  </Icon>
);

const ArrowDownRight = ({ className }) => (
  <Icon className={className}>
    <line x1="7" y1="7" x2="17" y2="17"></line>
    <polyline points="17 7 17 17 7 17"></polyline>
  </Icon>
);

const Building2 = ({ className }) => (
  <Icon className={className}>
    <path d="M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"></path>
    <path d="M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"></path>
    <path d="M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"></path>
    <path d="M10 6h4"></path>
    <path d="M10 10h4"></path>
    <path d="M10 14h4"></path>
    <path d="M10 18h4"></path>
  </Icon>
);

const Download = ({ className }) => (
  <Icon className={className}>
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
    <polyline points="7 10 12 15 17 10"></polyline>
    <line x1="12" y1="15" x2="12" y2="3"></line>
  </Icon>
);

const Printer = ({ className }) => (
  <Icon className={className}>
    <polyline points="6 9 6 2 18 2 18 9"></polyline>
    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
    <rect x="6" y="14" width="12" height="8"></rect>
  </Icon>
);

const Search = ({ className }) => (
  <Icon className={className}>
    <circle cx="11" cy="11" r="8"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
  </Icon>
);

const User = ({ className }) => (
  <Icon className={className}>
    <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
    <circle cx="12" cy="7" r="4"></circle>
  </Icon>
);

const Hash = ({ className }) => (
  <Icon className={className}>
    <line x1="4" y1="9" x2="20" y2="9"></line>
    <line x1="4" y1="15" x2="20" y2="15"></line>
    <line x1="10" y1="3" x2="8" y2="21"></line>
    <line x1="16" y1="3" x2="14" y2="21"></line>
  </Icon>
);

const CalendarRange = ({ className }) => (
  <Icon className={className}>
    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
    <line x1="16" y1="2" x2="16" y2="6"></line>
    <line x1="8" y1="2" x2="8" y2="6"></line>
    <line x1="3" y1="10" x2="21" y2="10"></line>
    <rect x="8" y="14" width="8" height="4"></rect>
  </Icon>
);

const Clock = ({ className }) => (
  <Icon className={className}>
    <circle cx="12" cy="12" r="10"></circle>
    <polyline points="12 6 12 12 16 14"></polyline>
  </Icon>
);

const Sparkles = ({ className }) => (
  <Icon className={className}>
    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path>
  </Icon>
);

const API_BASE_URL = "http://localhost:5000/api";

const EnhancedBankingDashboard = () => {
  // State management
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const [refreshing, setRefreshing] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  // Data states
  const [summary, setSummary] = useState(null);
  const [visualizations, setVisualizations] = useState(null);
  const [neuralNetworkResults, setNeuralNetworkResults] = useState(null);
  const [loadingNN, setLoadingNN] = useState(false);

  // Bank statement states
  const [availableBanks, setAvailableBanks] = useState([]);
  const [availableAccounts, setAvailableAccounts] = useState([]);
  const [filteredAccounts, setFilteredAccounts] = useState([]);
  const [accountSearchTerm, setAccountSearchTerm] = useState("");
  const [statementForm, setStatementForm] = useState({
    bank_name: "",
    account_number: "",
    from_date: "2025-05-01",
    to_date: "2025-05-31",
  });
  const [bankStatement, setBankStatement] = useState(null);
  const [loadingStatement, setLoadingStatement] = useState(false);
  const [loadingAccounts, setLoadingAccounts] = useState(false);

  // Colors for charts
  const COLORS = [
    "#3b82f6",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#84cc16",
    "#f97316",
    "#6366f1",
    "#14b8a6",
    "#f43f5e",
  ];

  // Initial loading effect
  useEffect(() => {
    const timer = setTimeout(() => {
      setInitialLoading(false);
    }, 3000);
    return () => clearTimeout(timer);
  }, []);

  // Fetch wrapper function with better error handling
  const fetchAPI = async (url, options = {}) => {
    try {
      console.log(`Fetching: ${url}`);
      const response = await fetch(url, {
        ...options,
        headers: {
          "Content-Type": "application/json",
          ...options.headers,
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`HTTP ${response.status}: ${errorText}`);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log(`Success: ${url}`, data.success);
      return data;
    } catch (error) {
      console.error("Fetch error:", error);
      throw error;
    }
  };

  // Fetch available banks
  const fetchAvailableBanks = useCallback(async () => {
    try {
      const response = await fetchAPI(`${API_BASE_URL}/available-banks`);
      if (response.success) {
        setAvailableBanks(response.banks);
      }
    } catch (err) {
      console.error("Error fetching banks:", err);
    }
  }, []);

  // Fetch available accounts for selected bank
  const fetchAvailableAccounts = useCallback(async (bankName) => {
    if (!bankName) {
      setAvailableAccounts([]);
      setFilteredAccounts([]);
      return;
    }

    setLoadingAccounts(true);
    try {
      const response = await fetchAPI(`${API_BASE_URL}/available-accounts`, {
        method: "POST",
        body: JSON.stringify({ sender_bank_name: bankName }),
      });
      if (response.success) {
        setAvailableAccounts(response.accounts);
        setFilteredAccounts(response.accounts);
      } else {
        setAvailableAccounts([]);
        setFilteredAccounts([]);
      }
    } catch (err) {
      console.error("Error fetching accounts:", err);
      setAvailableAccounts([]);
      setFilteredAccounts([]);
    } finally {
      setLoadingAccounts(false);
    }
  }, []);

  // Filter accounts based on search term
  const handleAccountSearch = (searchTerm) => {
    setAccountSearchTerm(searchTerm);
    if (!searchTerm.trim()) {
      setFilteredAccounts(availableAccounts);
      return;
    }

    const filtered = availableAccounts.filter((account) =>
      account.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredAccounts(filtered);
  };

  // Handle bank selection
  const handleBankSelection = (bankName) => {
    setStatementForm((prev) => ({
      ...prev,
      bank_name: bankName,
      account_number: "",
    }));
    setAccountSearchTerm("");
    fetchAvailableAccounts(bankName);
  };

  // Handle account selection
  const handleAccountSelection = (accountNumber) => {
    setStatementForm((prev) => ({ ...prev, account_number: accountNumber }));
    setAccountSearchTerm(accountNumber);
  };

  // Fetch all data with better error handling
  const fetchAllData = useCallback(async () => {
    if (!dataLoaded) return;

    setRefreshing(true);
    setError(null);

    try {
      console.log("Fetching all dashboard data...");

      const summaryRes = await fetchAPI(`${API_BASE_URL}/data-summary`);
      if (summaryRes.success) {
        setSummary(summaryRes.summary);
      }

      const vizRes = await fetchAPI(`${API_BASE_URL}/visualizations`);
      if (vizRes.success) {
        setVisualizations(vizRes.visualizations);
      }

      await fetchAvailableBanks();

      console.log("All data fetched successfully");
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(`Failed to fetch data: ${err.message}`);
    } finally {
      setRefreshing(false);
    }
  }, [dataLoaded, fetchAvailableBanks]);

  // Load data function
  const loadData = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log("Loading banking data...");
      const response = await fetchAPI(`${API_BASE_URL}/load-data`, {
        method: "POST",
      });

      if (response.success) {
        console.log("Data loaded successfully");
        setDataLoaded(true);
        setTimeout(() => {
          fetchAllData();
        }, 1000);
      } else {
        setError(response.error || "Failed to load data");
      }
    } catch (err) {
      console.error("Load data error:", err);
      setError(`Failed to load data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Generate bank statement
  const generateBankStatement = async () => {
    setLoadingStatement(true);
    setError(null);

    try {
      const response = await fetchAPI(`${API_BASE_URL}/bank-statement`, {
        method: "POST",
        body: JSON.stringify(statementForm),
      });

      if (response.success) {
        setBankStatement(response.statement);
      } else {
        setError(response.error || "Failed to generate statement");
      }
    } catch (err) {
      console.error("Statement generation error:", err);
      setError(`Failed to generate statement: ${err.message}`);
    } finally {
      setLoadingStatement(false);
    }
  };

  // Download PDF function
  const downloadStatementPDF = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/download-statement-pdf`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(statementForm),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `bank_statement_${statementForm.account_number}_${statementForm.from_date}_to_${statementForm.to_date}.pdf`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } else {
        setError("Failed to download PDF");
      }
    } catch (err) {
      console.error("PDF download error:", err);
      setError(`Failed to download PDF: ${err.message}`);
    }
  };

  const runNeuralNetworkAnalysis = async () => {
    setLoadingNN(true);
    setError(null);

    try {
      console.log("Running neural network analysis...");
      const response = await fetchAPI(
        `${API_BASE_URL}/neural-network-analysis`,
        {
          method: "POST",
        }
      );

      if (response.success) {
        setNeuralNetworkResults(response);
        console.log("Neural network analysis completed");
      } else {
        setError(response.error || "Neural network analysis failed");
      }
    } catch (err) {
      console.error("Neural network error:", err);
      setError(`Neural network analysis failed: ${err.message}`);
    } finally {
      setLoadingNN(false);
    }
  };

  // Auto-refresh data every 30 seconds
  useEffect(() => {
    if (dataLoaded) {
      const interval = setInterval(fetchAllData, 30000);
      return () => clearInterval(interval);
    }
  }, [dataLoaded, fetchAllData]);

  // Initialize filtered accounts when available accounts change
  useEffect(() => {
    setFilteredAccounts(availableAccounts);
  }, [availableAccounts]);

  // Format currency
  const formatCurrency = (value) => {
    if (value == null || isNaN(value)) return "₹0";
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Format number
  const formatNumber = (value) => {
    if (value == null || isNaN(value)) return "0";
    return new Intl.NumberFormat("en-IN").format(value);
  };

  // Calculate percentage change
  const calculatePercentageChange = (current, previous) => {
    if (previous === 0 || !current || !previous) return 0;
    return (((current - previous) / previous) * 100).toFixed(2);
  };

  // Custom tooltip for charts
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white/95 backdrop-blur-sm p-2.5 rounded-lg shadow-xl border border-gray-200/50">
          <p className="text-gray-600 text-xs font-medium mb-1">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-xs" style={{ color: entry.color }}>
              {entry.name}:{" "}
              {typeof entry.value === "number" && entry.value > 1000
                ? formatCurrency(entry.value)
                : formatNumber(entry.value)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  // Glass Card Component with improved design
  const GlassCard = ({ children, className = "", hover = true }) => (
    <div
      className={`
        relative overflow-hidden
        bg-white/60 backdrop-blur-md
        border border-white/30
        rounded-xl shadow-md
        ${hover ? "hover:shadow-lg hover:bg-white/70 hover:scale-[1.01]" : ""}
        transition-all duration-300 ease-out
        ${className}
      `}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent opacity-40"></div>
      <div className="relative z-10">{children}</div>
    </div>
  );

  // Animated Stats Card Component with improved styling
  const StatsCard = ({ title, value, change, icon: Icon, color, subtitle }) => {
    const isPositive = change >= 0;

    const gradientClasses = {
      blue: "from-blue-500/10 to-blue-600/10",
      green: "from-green-500/10 to-green-600/10",
      amber: "from-amber-500/10 to-amber-600/10",
      red: "from-red-500/10 to-red-600/10",
      purple: "from-purple-500/10 to-purple-600/10",
      cyan: "from-cyan-500/10 to-cyan-600/10",
    };

    const iconGradients = {
      blue: "from-blue-500 to-blue-600",
      green: "from-green-500 to-green-600",
      amber: "from-amber-500 to-amber-600",
      red: "from-red-500 to-red-600",
      purple: "from-purple-500 to-purple-600",
      cyan: "from-cyan-500 to-cyan-600",
    };

    return (
      <GlassCard className="p-3.5 group cursor-pointer">
        <div
          className={`absolute inset-0 bg-gradient-to-br ${gradientClasses[color]} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}
        ></div>
        <div className="relative flex items-center justify-between">
          <div className="flex-1">
            <p className="text-gray-600 text-xs font-medium mb-1">{title}</p>
            <p className="text-lg font-bold text-gray-900 mb-0.5">{value}</p>
            {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
            {change !== undefined && !isNaN(change) && (
              <div
                className={`flex items-center mt-1.5 text-xs ${
                  isPositive ? "text-green-600" : "text-red-600"
                }`}
              >
                {isPositive ? (
                  <ArrowUpRight className="w-3 h-3 mr-1" />
                ) : (
                  <ArrowDownRight className="w-3 h-3 mr-1" />
                )}
                <span className="font-medium">{Math.abs(change)}%</span>
                <span className="text-gray-500 ml-1 text-xs">
                  vs last month
                </span>
              </div>
            )}
          </div>
          <div
            className={`w-9 h-9 rounded-lg bg-gradient-to-br ${iconGradients[color]} flex items-center justify-center shadow-md group-hover:scale-110 transition-transform duration-300`}
          >
            <Icon className="w-4 h-4 text-white" />
          </div>
        </div>
      </GlassCard>
    );
  };

  // Enhanced Tab Button Component with improved design
  const TabButton = ({ id, label, icon: Icon, active, onClick }) => {
    const tabColors = {
      overview: "blue",
      visualizations: "green",
      "neural-network": "purple",
      "bank-statement": "amber",
    };

    const activeColor = tabColors[id] || "blue";
    const gradientClasses = {
      blue: "from-blue-500 to-blue-600",
      green: "from-green-500 to-green-600",
      amber: "from-amber-500 to-amber-600",
      purple: "from-purple-500 to-purple-600",
    };

    return (
      <button
        onClick={() => onClick(id)}
        className={`
          relative flex items-center px-3.5 py-1.5 rounded-lg font-medium text-xs
          transition-all duration-300 ease-in-out overflow-hidden group
          ${
            active
              ? `bg-gradient-to-r ${gradientClasses[activeColor]} text-white shadow-md`
              : "bg-white/60 backdrop-blur-sm text-gray-600 hover:bg-white/80 border border-white/20"
          }
        `}
      >
        {!active && (
          <div
            className={`absolute inset-0 bg-gradient-to-r ${gradientClasses[activeColor]} opacity-0 group-hover:opacity-8 transition-opacity duration-300`}
          ></div>
        )}
        <Icon className="w-3.5 h-3.5 mr-1.5 relative z-10" />
        <span className="relative z-10">{label}</span>
        {active && (
          <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white/30"></div>
        )}
      </button>
    );
  };

  // Error Display Component with improved styling
  const ErrorDisplay = ({ error, onRetry }) => (
    <GlassCard className="p-3.5 mb-4 border-red-200/50 bg-red-50/70">
      <div className="flex items-center">
        <AlertCircle className="w-4 h-4 text-red-500 mr-2 flex-shrink-0" />
        <div className="flex-1">
          <h3 className="text-red-800 font-medium text-xs">Error</h3>
          <p className="text-red-700 text-xs mt-0.5">{error}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="ml-4 px-2.5 py-1 bg-red-100/70 text-red-700 rounded-lg hover:bg-red-200/70 transition-colors text-xs"
          >
            Retry
          </button>
        )}
      </div>
    </GlassCard>
  );

  // Enhanced Loading Screen Component with Aceternity UI inspired animations
  const LoadingScreen = () => (
    <div className="fixed inset-0 bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center z-50">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full filter blur-3xl opacity-70 animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full filter blur-3xl opacity-70 animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-cyan-500/10 rounded-full filter blur-3xl opacity-70 animate-pulse delay-500"></div>

        {/* Flying particles */}
        <div className="absolute inset-0">
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full animate-float"
              style={{
                top: `${Math.random() * 100}%`,
                left: `${Math.random() * 100}%`,
                opacity: Math.random() * 0.5 + 0.3,
                animationDuration: `${Math.random() * 10 + 10}s`,
                animationDelay: `${Math.random() * 5}s`,
              }}
            ></div>
          ))}
        </div>
      </div>

      <div className="relative z-10 text-center">
        <div className="relative mb-8">
          {/* Bank logo with animated elements */}
          <div className="w-20 h-20 mx-auto relative">
            <div className="absolute inset-0 border-4 border-blue-500/30 rounded-full animate-spin-slow"></div>
            <div className="absolute inset-2 border-4 border-purple-500/30 rounded-full animate-spin-reverse"></div>
            <div className="absolute inset-4 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg shadow-blue-500/20">
              <Building2 className="w-6 h-6 text-white animate-pulse" />
            </div>
          </div>

          {/* Sparkles */}
          <div className="absolute top-0 left-0 w-full h-full">
            <Sparkles className="absolute top-2 right-6 w-3 h-3 text-blue-300 animate-ping" />
            <Sparkles className="absolute bottom-4 left-8 w-2 h-2 text-purple-300 animate-ping delay-300" />
            <Sparkles className="absolute top-8 left-4 w-2 h-2 text-cyan-300 animate-ping delay-700" />
          </div>
        </div>

        <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 mb-3">
          FintelliGen
        </h1>
        <p className="text-gray-300 text-sm mb-6 animate-pulse">
          Initializing financial intelligence...
        </p>

        {/* Loading bar with pulse animation */}
        <div className="w-48 h-1 mx-auto bg-gray-700/50 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-500 rounded-full animate-loading-bar"></div>
        </div>
      </div>
    </div>
  );

  // Format non-transaction periods for display
  const formatNonTransactionPeriods = (periods, customerName) => {
    if (!periods || periods.length === 0) return null;

    return periods
      .map((period, index) => {
        const [startDate, endDate] = period;
        const start = new Date(startDate).toLocaleDateString("en-GB");
        const end = new Date(endDate).toLocaleDateString("en-GB");

        if (start === end) {
          return `On ${start}, ${customerName} has not made any transactions.`;
        } else {
          return `From ${start} to ${end}, ${customerName} has not made any transactions.`;
        }
      })
      .join(" ");
  };

  // Bank Statement Component with improved styling
  const BankStatementView = () => {
    if (!bankStatement) {
      return (
        <GlassCard className="p-5">
          <div className="max-w-4xl mx-auto">
            {/* Header Section */}
            <div className="text-center mb-5">
              <div className="flex items-center justify-center mb-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mr-3 shadow-md">
                  <Building2 className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-gray-900">
                    Bank Statement Generator
                  </h2>
                  <p className="text-gray-600 text-xs">
                    Generate professional bank statements for multiple banks
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              {/* Bank Selection */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  <Building2 className="w-3.5 h-3.5 inline mr-1.5" />
                  Select Bank
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-2.5">
                  {availableBanks.map((bank) => (
                    <button
                      key={bank}
                      onClick={() => handleBankSelection(bank)}
                      className={`p-2.5 rounded-lg border-2 transition-all duration-300 text-left hover:scale-[1.02] ${
                        statementForm.bank_name === bank
                          ? "border-blue-500 bg-blue-50/70 shadow-md backdrop-blur-sm"
                          : "border-gray-200/50 hover:border-gray-300/50 hover:bg-gray-50/70 bg-white/50 backdrop-blur-sm"
                      }`}
                    >
                      <div className="flex items-center">
                        <div className="w-7 h-7 bg-blue-100 rounded-lg flex items-center justify-center mr-2.5">
                          <Building2 className="w-3.5 h-3.5 text-blue-600" />
                        </div>
                        <div>
                          <div className="font-medium text-gray-900 text-xs">
                            {bank}
                          </div>
                          <div className="text-[10px] text-gray-500">
                            Select Bank
                          </div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Account Number Selection */}
              {statementForm.bank_name && (
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-2">
                    <User className="w-3.5 h-3.5 inline mr-1.5" />
                    Select Account Number
                  </label>

                  {/* Search Box for Accounts */}
                  {availableAccounts.length > 0 && (
                    <div className="mb-3">
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
                        <input
                          type="text"
                          placeholder="Search account numbers..."
                          value={accountSearchTerm}
                          onChange={(e) => handleAccountSearch(e.target.value)}
                          className="w-full pl-9 pr-3 py-2 border border-gray-300/50 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white/70 backdrop-blur-sm text-xs"
                        />
                      </div>
                    </div>
                  )}

                  {loadingAccounts ? (
                    <div className="flex items-center justify-center py-5">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-600 mr-2" />
                      <span className="text-gray-600 text-xs">
                        Loading accounts...
                      </span>
                    </div>
                  ) : filteredAccounts.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2.5">
                      {filteredAccounts.map((account) => (
                        <button
                          key={account}
                          onClick={() => handleAccountSelection(account)}
                          className={`p-2.5 rounded-lg border-2 transition-all duration-300 text-left hover:scale-[1.02] ${
                            statementForm.account_number === account
                              ? "border-green-500 bg-green-50/70 shadow-md backdrop-blur-sm"
                              : "border-gray-200/50 hover:border-gray-300/50 hover:bg-gray-50/70 bg-white/50 backdrop-blur-sm"
                          }`}
                        >
                          <div className="flex items-center">
                            <div className="w-7 h-7 bg-green-100 rounded-lg flex items-center justify-center mr-2.5">
                              <Hash className="w-3.5 h-3.5 text-green-600" />
                            </div>
                            <div>
                              <div className="font-medium text-gray-900 text-xs">
                                {account}
                              </div>
                              <div className="text-[10px] text-gray-500">
                                Account Number
                              </div>
                            </div>
                          </div>
                        </button>
                      ))}
                    </div>
                  ) : accountSearchTerm && availableAccounts.length > 0 ? (
                    <div className="text-center py-5 text-gray-500">
                      <Search className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                      <p className="text-xs">
                        No accounts found matching "{accountSearchTerm}"
                      </p>
                      <button
                        onClick={() => handleAccountSearch("")}
                        className="mt-2 text-blue-600 hover:text-blue-700 text-xs"
                      >
                        Clear search
                      </button>
                    </div>
                  ) : (
                    <div className="text-center py-5 text-gray-500">
                      <User className="w-8 h-8 mx-auto mb-2 text-gray-300" />
                      <p className="text-xs">
                        No accounts found for {statementForm.bank_name}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Date Range Selection */}
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  <CalendarRange className="w-3.5 h-3.5 inline mr-1.5" />
                  Select Date Range
                </label>

                {/* Quick Date Presets */}
                <div className="mb-3">
                  <label className="block text-[10px] text-gray-600 mb-1.5">
                    Quick Select:
                  </label>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      onClick={() =>
                        setStatementForm((prev) => ({
                          ...prev,
                          from_date: "2025-05-01",
                          to_date: "2025-05-31",
                        }))
                      }
                      className={`px-2.5 py-1 rounded-lg text-[10px] transition-all duration-300 hover:scale-105 ${
                        statementForm.from_date === "2025-05-01" &&
                        statementForm.to_date === "2025-05-30"
                          ? "bg-green-500 text-white shadow-md"
                          : "bg-green-100/70 text-green-700 hover:bg-green-200/70 backdrop-blur-sm"
                      }`}
                    >
                      May 2025
                    </button>
                    <button
                      onClick={() =>
                        setStatementForm((prev) => ({
                          ...prev,
                          from_date: "2025-06-01",
                          to_date: "2025-06-31",
                        }))
                      }
                      className={`px-2.5 py-1 rounded-lg text-[10px] transition-all duration-300 hover:scale-105 ${
                        statementForm.from_date === "2025-06-01" &&
                        statementForm.to_date === "2025-06-30"
                          ? "bg-green-500 text-white shadow-md"
                          : "bg-green-100/70 text-green-700 hover:bg-green-200/70 backdrop-blur-sm"
                      }`}
                    >
                      June 2025
                    </button>
                  </div>
                </div>

                {/* Custom Date Range */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-[10px] text-gray-600 mb-1.5">
                      From Date
                    </label>
                    <input
                      type="date"
                      value={statementForm.from_date}
                      onChange={(e) =>
                        setStatementForm((prev) => ({
                          ...prev,
                          from_date: e.target.value,
                        }))
                      }
                      className="w-full p-2 border border-gray-300/50 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white/70 backdrop-blur-sm text-xs"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-gray-600 mb-1.5">
                      To Date
                    </label>
                    <input
                      type="date"
                      value={statementForm.to_date}
                      onChange={(e) =>
                        setStatementForm((prev) => ({
                          ...prev,
                          to_date: e.target.value,
                        }))
                      }
                      className="w-full p-2 border border-gray-300/50 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white/70 backdrop-blur-sm text-xs"
                    />
                  </div>
                </div>
              </div>

              {/* Generate Button */}
              <div className="pt-3">
                <button
                  onClick={generateBankStatement}
                  disabled={
                    !statementForm.bank_name ||
                    !statementForm.account_number ||
                    !statementForm.from_date ||
                    !statementForm.to_date ||
                    loadingStatement
                  }
                  className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-2.5 px-5 rounded-lg font-medium hover:from-blue-700 hover:to-blue-800 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-md hover:shadow-lg transform hover:scale-[1.02] text-xs"
                >
                  {loadingStatement ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 mr-2 animate-spin" />
                      Generating Statement...
                    </>
                  ) : (
                    <>
                      <Search className="w-3.5 h-3.5 mr-2" />
                      Generate Bank Statement
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </GlassCard>
      );
    }

    // Display generated statement
    return (
      <GlassCard className="overflow-hidden">
        {/* Statement Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="w-9 h-9 bg-white/20 rounded-lg flex items-center justify-center mr-3">
                <Building2 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-bold">
                  {bankStatement.bank_info.name}
                </h2>
                <p className="text-blue-100 text-xs">Account Statement</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => window.print()}
                className="bg-white/20 hover:bg-white/30 text-white px-2.5 py-1 rounded-lg transition-colors flex items-center text-xs"
              >
                <Printer className="w-3 h-3 mr-1" />
                Print
              </button>
              <button
                onClick={downloadStatementPDF}
                className="bg-white/20 hover:bg-white/30 text-white px-2.5 py-1 rounded-lg transition-colors flex items-center text-xs"
              >
                <Download className="w-3 h-3 mr-1" />
                Download PDF
              </button>
            </div>
          </div>
        </div>

        <div className="p-4">
          {/* Account Details and Statement Period */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
            {/* Account Details */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-2.5 text-sm">
                Account Details
              </h3>
              <div className="space-y-1.5">
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    Account Holder:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.account_details.account_holder}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    Account Number:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.account_details.account_number}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    Account Type:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.account_details.account_type}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    IFSC Code:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.account_details.ifsc_code}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-gray-600 font-medium text-xs">
                    Branch:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.account_details.branch}
                  </span>
                </div>
              </div>
            </div>

            {/* Statement Period */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-2.5 text-sm">
                Statement Period
              </h3>
              <div className="space-y-1.5">
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    From Date:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.statement_period.from_date}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    To Date:
                  </span>
                  <span className="font-semibold text-gray-900 text-xs">
                    {bankStatement.statement_period.to_date}
                  </span>
                </div>
                <div className="flex justify-between py-1 border-b border-gray-100">
                  <span className="text-gray-600 font-medium text-xs">
                    Opening Balance:
                  </span>
                  <span className="font-semibold text-green-600 text-xs">
                    {formatCurrency(
                      bankStatement.statement_period.opening_balance
                    )}
                  </span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-gray-600 font-medium text-xs">
                    Closing Balance:
                  </span>
                  <span className="font-semibold text-blue-600 text-xs">
                    {formatCurrency(
                      bankStatement.statement_period.closing_balance
                    )}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-2.5 mb-5">
            <div className="bg-green-50/70 border border-green-200/50 rounded-lg p-2.5 text-center backdrop-blur-sm">
              <div className="text-base font-bold text-green-600 mb-0.5">
                {formatCurrency(bankStatement.summary.total_credits)}
              </div>
              <div className="text-[10px] text-green-700 font-medium flex items-center justify-center">
                <TrendingUp className="w-2.5 h-2.5 mr-1" />
                Total Credits
              </div>
            </div>
            <div className="bg-red-50/70 border border-red-200/50 rounded-lg p-2.5 text-center backdrop-blur-sm">
              <div className="text-base font-bold text-red-600 mb-0.5">
                {formatCurrency(bankStatement.summary.total_debits)}
              </div>
              <div className="text-[10px] text-red-700 font-medium flex items-center justify-center">
                <TrendingDown className="w-2.5 h-2.5 mr-1" />
                Total Debits
              </div>
            </div>
            <div className="bg-blue-50/70 border border-blue-200/50 rounded-lg p-2.5 text-center backdrop-blur-sm">
              <div className="text-base font-bold text-blue-600 mb-0.5">
                {bankStatement.summary.transaction_count}
              </div>
              <div className="text-[10px] text-blue-700 font-medium flex items-center justify-center">
                <Activity className="w-2.5 h-2.5 mr-1" />
                Transactions
              </div>
            </div>
            <div className="bg-purple-50/70 border border-purple-200/50 rounded-lg p-2.5 text-center backdrop-blur-sm">
              <div
                className={`text-base font-bold mb-0.5 ${
                  bankStatement.summary.net_change >= 0
                    ? "text-green-600"
                    : "text-red-600"
                }`}
              >
                {formatCurrency(bankStatement.summary.net_change)}
              </div>
              <div className="text-[10px] text-purple-700 font-medium flex items-center justify-center">
                <DollarSign className="w-2.5 h-2.5 mr-1" />
                Net Change
              </div>
            </div>
            <div className="bg-gray-50/70 border border-gray-200/50 rounded-lg p-2.5 text-center backdrop-blur-sm">
              <div className="text-base font-bold text-gray-600 mb-0.5">
                {bankStatement.summary.total_days}
              </div>
              <div className="text-[10px] text-gray-700 font-medium flex items-center justify-center">
                <Clock className="w-2.5 h-2.5 mr-1" />
                Total Days
              </div>
              <div className="text-[10px] text-gray-500 mt-0.5">
                {bankStatement.summary.transaction_days} active,{" "}
                {bankStatement.summary.non_transaction_days} no activity
              </div>
            </div>
          </div>

          {/* Transaction Details */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-2.5">
              Transaction Details
            </h3>
            {/* Transaction table with only actual transactions */}
            <div className="overflow-x-auto border border-gray-200/50 rounded-lg backdrop-blur-sm">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50/70 backdrop-blur-sm">
                  <tr>
                    <th className="px-2.5 py-1.5 text-left text-[10px] font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                      Date
                    </th>
                    <th className="px-2.5 py-1.5 text-left text-[10px] font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                      Description
                    </th>
                    <th className="px-2.5 py-1.5 text-left text-[10px] font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                      Reference No.
                    </th>
                    <th className="px-2.5 py-1.5 text-right text-[10px] font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                      Debit (₹)
                    </th>
                    <th className="px-2.5 py-1.5 text-right text-[10px] font-medium text-gray-500 uppercase tracking-wider border-r border-gray-200">
                      Credit (₹)
                    </th>
                    <th className="px-2.5 py-1.5 text-right text-[10px] font-medium text-gray-500 uppercase tracking-wider">
                      Balance (₹)
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white/70 divide-y divide-gray-200 backdrop-blur-sm">
                  {bankStatement.transactions.length > 0 ? (
                    bankStatement.transactions.map((transaction, index) => (
                      <tr
                        key={index}
                        className="hover:bg-gray-50/70 transition-colors"
                      >
                        <td className="px-2.5 py-1.5 whitespace-nowrap text-[10px] border-r border-gray-100">
                          <div className="font-medium text-gray-900 flex items-center">
                            {transaction.date}
                          </div>
                          <div className="text-gray-500 text-[10px]">
                            {transaction.time}
                          </div>
                        </td>
                        <td className="px-2.5 py-1.5 text-[10px] border-r border-gray-100">
                          <div className="max-w-xs text-gray-900">
                            {transaction.description}
                          </div>
                        </td>
                        <td className="px-2.5 py-1.5 whitespace-nowrap text-[10px] text-gray-500 border-r border-gray-100 font-mono">
                          {transaction.reference_no}
                        </td>
                        <td className="px-2.5 py-1.5 whitespace-nowrap text-[10px] text-right border-r border-gray-100">
                          {transaction.debit ? (
                            <span className="text-red-600 font-semibold">
                              {formatCurrency(transaction.debit)}
                            </span>
                          ) : (
                            <span className="text-gray-400">—</span>
                          )}
                        </td>
                        <td className="px-2.5 py-1.5 whitespace-nowrap text-[10px] text-right border-r border-gray-100">
                          {transaction.credit ? (
                            <span className="text-green-600 font-semibold">
                              {formatCurrency(transaction.credit)}
                            </span>
                          ) : (
                            <span className="text-gray-400">—</span>
                          )}
                        </td>
                        <td className="px-2.5 py-1.5 whitespace-nowrap text-[10px] text-right">
                          <span
                            className={`font-semibold ${
                              transaction.balance >= 0
                                ? "text-blue-600"
                                : "text-red-600"
                            }`}
                          >
                            {formatCurrency(transaction.balance)}
                          </span>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td
                        colSpan="6"
                        className="px-2.5 py-8 text-center text-gray-500 text-xs"
                      >
                        No transactions found for the selected period
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Non-transaction periods summary below the table */}
          {bankStatement.non_transaction_periods &&
            bankStatement.non_transaction_periods.length > 0 && (
              <div className="mt-4 p-3 bg-yellow-50/70 border border-yellow-200/50 rounded-lg backdrop-blur-sm">
                <h4 className="text-xs font-semibold text-yellow-800 mb-2 flex items-center">
                  <Clock className="w-3 h-3 mr-1.5" />
                  Non-Transaction Periods
                </h4>
                <p className="text-xs text-yellow-700">
                  {formatNonTransactionPeriods(
                    bankStatement.non_transaction_periods,
                    bankStatement.customer_name
                  )}
                </p>
              </div>
            )}

          {/* Footer */}
          <div className="mt-5 pt-3 border-t border-gray-200 text-center text-[10px] text-gray-500 space-y-0.5">
            <p className="font-medium">
              This is a computer-generated statement and does not require a
              signature.
            </p>
            <p>
              For any queries, please contact your nearest branch or call
              customer service.
            </p>
            <p className="text-[9px]">
              Generated on:{" "}
              {new Date().toLocaleDateString("en-GB", {
                day: "2-digit",
                month: "long",
                year: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </p>
          </div>

          {/* Back Button */}
          <div className="mt-4 text-center">
            <button
              onClick={() => setBankStatement(null)}
              className="bg-gray-600 text-white px-4 py-1.5 rounded-lg hover:bg-gray-700 transition-colors inline-flex items-center text-xs"
            >
              ← Back to Configuration
            </button>
          </div>
        </div>
      </GlassCard>
    );
  };

  // Initial loading screen
  if (initialLoading) {
    return <LoadingScreen />;
  }

  if (!dataLoaded) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 flex items-center justify-center relative overflow-hidden">
        {/* Animated background elements */}
        <div className="absolute inset-0">
          <div className="absolute top-20 left-20 w-64 h-64 bg-blue-500/10 rounded-full filter blur-xl animate-pulse"></div>
          <div className="absolute bottom-20 right-20 w-80 h-80 bg-purple-500/10 rounded-full filter blur-xl animate-pulse delay-1000"></div>
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-cyan-500/10 rounded-full filter blur-xl animate-pulse delay-500"></div>
        </div>

        <GlassCard className="p-5 max-w-md w-full mx-4">
          <div className="text-center">
            <div className="w-14 h-14 bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-3 shadow-md">
              <BarChart3 className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-xl font-bold text-gray-900 mb-1.5">
              FintelliGen
            </h1>
            <p className="text-gray-600 mb-4 text-xs">
              Load your banking dataset to start analysis
            </p>

            {error && <ErrorDisplay error={error} onRetry={loadData} />}

            <button
              onClick={loadData}
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 text-white py-2.5 px-5 rounded-lg font-medium hover:from-blue-700 hover:to-blue-800 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-md hover:shadow-lg transform hover:scale-[1.02] text-xs"
            >
              {loading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 mr-2 animate-spin" />
                  Loading Banking Data...
                </>
              ) : (
                <>
                  <FileText className="w-3.5 h-3.5 mr-2" />
                  Load Banking Data
                </>
              )}
            </button>
          </div>
        </GlassCard>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-20 w-64 h-64 bg-blue-500/5 rounded-full filter blur-xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-80 h-80 bg-purple-500/5 rounded-full filter blur-xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-cyan-500/5 rounded-full filter blur-xl animate-pulse delay-500"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 backdrop-blur-sm bg-white/70 shadow-sm border-b border-white/20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="w-7 h-7 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center mr-2.5 shadow-md">
                <BarChart3 className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold text-gray-900">
                  FintelliGen
                </h1>
                <p className="text-[10px] text-gray-600">
                  Real-time Banking Intelligence & Analytics
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2.5">
              <div className="flex items-center text-[10px] text-gray-600">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full mr-1.5 animate-pulse"></div>
                Online
              </div>
              <button
                onClick={fetchAllData}
                disabled={refreshing}
                className="flex items-center px-2.5 py-1 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all duration-300 disabled:opacity-50 shadow-md hover:shadow-lg transform hover:scale-[1.02] text-xs"
              >
                <RefreshCw
                  className={`w-2.5 h-2.5 mr-1 ${
                    refreshing ? "animate-spin" : ""
                  }`}
                />
                {refreshing ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
        <div className="flex flex-wrap gap-1.5">
          <TabButton
            id="overview"
            label="Overview"
            icon={BarChart3}
            active={activeTab === "overview"}
            onClick={setActiveTab}
          />
          <TabButton
            id="visualizations"
            label="Analytics"
            icon={Activity}
            active={activeTab === "visualizations"}
            onClick={setActiveTab}
          />
          <TabButton
            id="neural-network"
            label="AI Analysis"
            icon={Brain}
            active={activeTab === "neural-network"}
            onClick={setActiveTab}
          />
          <TabButton
            id="bank-statement"
            label="Bank Statement"
            icon={FileText}
            active={activeTab === "bank-statement"}
            onClick={setActiveTab}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-6">
        {error && <ErrorDisplay error={error} onRetry={fetchAllData} />}

        {/* Overview Tab */}
        {activeTab === "overview" && (
          <div className="space-y-4">
            {summary ? (
              <>
                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  <StatsCard
                    title="Total Transactions"
                    value={formatNumber(summary.total_records)}
                    change={
                      summary.monthly_comparison
                        ? calculatePercentageChange(
                            summary.monthly_comparison.current_month.count,
                            summary.monthly_comparison.last_month.count
                          )
                        : 0
                    }
                    icon={Activity}
                    color="blue"
                    subtitle="All processed transactions"
                  />
                  <StatsCard
                    title="Total Volume"
                    value={formatCurrency(summary.total_amount)}
                    change={
                      summary.monthly_comparison
                        ? calculatePercentageChange(
                            summary.monthly_comparison.current_month
                              .total_amount,
                            summary.monthly_comparison.last_month.total_amount
                          )
                        : 0
                    }
                    icon={DollarSign}
                    color="green"
                    subtitle="Transaction value"
                  />
                  <StatsCard
                    title="Average Transaction"
                    value={formatCurrency(summary.average_transaction)}
                    change={
                      summary.monthly_comparison
                        ? calculatePercentageChange(
                            summary.monthly_comparison.current_month.avg_amount,
                            summary.monthly_comparison.last_month.avg_amount
                          )
                        : 0
                    }
                    icon={CreditCard}
                    color="amber"
                    subtitle="Per transaction"
                  />
                  <StatsCard
                    title="Date Range"
                    value="2025-05-01 to 2025-06-30"
                    icon={Calendar}
                    color="purple"
                    subtitle="Data period"
                  />
                </div>

                {/* Monthly Comparison Cards */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <GlassCard className="p-4">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                      <Calendar className="w-3.5 h-3.5 mr-1.5 text-amber-600" />
                      Last Month Summary
                    </h3>
                    <div className="space-y-2.5">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600 text-xs">
                          Transactions
                        </span>
                        <span className="font-medium text-xs">
                          {formatNumber(
                            summary.monthly_comparison?.last_month?.count || 0
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600 text-xs">
                          Total Amount
                        </span>
                        <span className="font-medium text-xs">
                          {formatCurrency(
                            summary.monthly_comparison?.last_month
                              ?.total_amount || 0
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600 text-xs">
                          Average Amount
                        </span>
                        <span className="font-medium text-xs">
                          {formatCurrency(
                            summary.monthly_comparison?.last_month
                              ?.avg_amount || 0
                          )}
                        </span>
                      </div>
                    </div>
                  </GlassCard>

                  <GlassCard className="p-4">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center">
                      <Calendar className="w-3.5 h-3.5 mr-1.5 text-blue-600" />
                      Current Month Summary
                    </h3>
                    <div className="space-y-2.5">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600 text-xs">
                          Transactions
                        </span>
                        <span className="font-medium text-xs">
                          {formatNumber(
                            summary.monthly_comparison?.current_month?.count ||
                              0
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600 text-xs">
                          Total Amount
                        </span>
                        <span className="font-medium text-xs">
                          {formatCurrency(
                            summary.monthly_comparison?.current_month
                              ?.total_amount || 0
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-gray-600 text-xs">
                          Average Amount
                        </span>
                        <span className="font-medium text-xs">
                          {formatCurrency(
                            summary.monthly_comparison?.current_month
                              ?.avg_amount || 0
                          )}
                        </span>
                      </div>
                    </div>
                  </GlassCard>
                </div>
              </>
            ) : (
              <GlassCard className="p-5 text-center">
                <Loader2 className="w-5 h-5 text-blue-600 mx-auto mb-2 animate-spin" />
                <p className="text-gray-600 text-xs">
                  Loading overview data...
                </p>
              </GlassCard>
            )}
          </div>
        )}

        {/* Visualizations Tab */}
        {activeTab === "visualizations" && (
          <div className="space-y-4">
            {visualizations ? (
              <>
                {/* Transaction Types and Status */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Transaction Type Distribution */}
                  {visualizations.transaction_type_pie && (
                    <GlassCard className="p-4">
                      <h3 className="text-sm font-semibold text-gray-900 mb-3">
                        Transaction Types
                      </h3>
                      <ResponsiveContainer width="100%" height={260}>
                        <PieChart>
                          <Pie
                            data={visualizations.transaction_type_pie.labels.map(
                              (label, index) => ({
                                name: label,
                                value:
                                  visualizations.transaction_type_pie.values[
                                    index
                                  ],
                              })
                            )}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) =>
                              `${name}: ${(percent * 100).toFixed(1)}%`
                            }
                            outerRadius={80}
                            fill="#8884d8"
                            dataKey="value"
                          >
                            {visualizations.transaction_type_pie.labels.map(
                              (entry, index) => (
                                <Cell
                                  key={`cell-${index}`}
                                  fill={COLORS[index % COLORS.length]}
                                />
                              )
                            )}
                          </Pie>
                          <Tooltip content={<CustomTooltip />} />
                        </PieChart>
                      </ResponsiveContainer>
                    </GlassCard>
                  )}

                  {/* Transaction Status Distribution */}
                  {visualizations.status_distribution && (
                    <GlassCard className="p-4">
                      <h3 className="text-sm font-semibold text-gray-900 mb-3">
                        Transaction Status Overview
                      </h3>
                      <ResponsiveContainer width="100%" height={260}>
                        <RadialBarChart
                          cx="50%"
                          cy="50%"
                          innerRadius="10%"
                          outerRadius="80%"
                          data={visualizations.status_distribution.labels.map(
                            (label, index) => ({
                              name: label,
                              value:
                                visualizations.status_distribution.values[
                                  index
                                ],
                              fill: COLORS[index % COLORS.length],
                            })
                          )}
                        >
                          <RadialBar
                            minAngle={15}
                            label={{ position: "insideStart", fill: "#fff" }}
                            background
                            clockWise
                            dataKey="value"
                          />
                          <Legend
                            iconSize={8}
                            layout="vertical"
                            verticalAlign="middle"
                            align="right"
                          />
                          <Tooltip />
                        </RadialBarChart>
                      </ResponsiveContainer>
                    </GlassCard>
                  )}
                </div>

                {/* Monthly Bar Chart */}
                <GlassCard className="p-4">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">
                    Monthly Transaction Volume
                  </h3>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart
                      data={visualizations.monthly_bar_chart.categories.map(
                        (cat, index) => ({
                          month: cat,
                          amount:
                            visualizations.monthly_bar_chart.total_amount[
                              index
                            ],
                          count:
                            visualizations.monthly_bar_chart.transaction_count[
                              index
                            ],
                        })
                      )}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        strokeOpacity={0.4}
                      />
                      <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend wrapperStyle={{ fontSize: "10px" }} />
                      <Bar
                        dataKey="amount"
                        fill="#3b82f6"
                        name="Total Amount"
                        radius={[6, 6, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </GlassCard>

                {/* Weekly Trend Analysis */}
                <GlassCard className="p-4">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">
                    Weekly Trend Analysis
                  </h3>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        strokeOpacity={0.4}
                      />
                      <XAxis dataKey="week" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend wrapperStyle={{ fontSize: "10px" }} />
                      {visualizations.weekly_trend.last_month.weeks.length >
                        0 && (
                        <Line
                          data={visualizations.weekly_trend.last_month.weeks.map(
                            (week, index) => ({
                              week,
                              amount:
                                visualizations.weekly_trend.last_month
                                  .total_amount[index],
                            })
                          )}
                          type="monotone"
                          dataKey="amount"
                          stroke="#ef4444"
                          strokeWidth={2}
                          name="Last Month"
                          dot={{ fill: "#ef4444", r: 3 }}
                          activeDot={{ r: 5 }}
                        />
                      )}
                      {visualizations.weekly_trend.current_month.weeks.length >
                        0 && (
                        <Line
                          data={visualizations.weekly_trend.current_month.weeks.map(
                            (week, index) => ({
                              week,
                              amount:
                                visualizations.weekly_trend.current_month
                                  .total_amount[index],
                            })
                          )}
                          type="monotone"
                          dataKey="amount"
                          stroke="#10b981"
                          strokeWidth={2}
                          name="Current Month"
                          dot={{ fill: "#10b981", r: 3 }}
                          activeDot={{ r: 5 }}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </GlassCard>

                {/* Debit/Credit Analysis */}
                <GlassCard className="p-4">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">
                    Debit vs Credit Analysis
                  </h3>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart
                      data={visualizations.debit_credit_analysis.categories.map(
                        (cat, index) => ({
                          month: cat,
                          debit:
                            visualizations.debit_credit_analysis.debit[index],
                          credit:
                            visualizations.debit_credit_analysis.credit[index],
                        })
                      )}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        strokeOpacity={0.4}
                      />
                      <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend wrapperStyle={{ fontSize: "10px" }} />
                      <Bar
                        dataKey="debit"
                        fill="#ef4444"
                        name="Debit"
                        radius={[6, 6, 0, 0]}
                      />
                      <Bar
                        dataKey="credit"
                        fill="#10b981"
                        name="Credit"
                        radius={[6, 6, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </GlassCard>

                {/* Balance Analysis */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <GlassCard className="p-4">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3">
                      Opening Balance Trend
                    </h3>
                    <ResponsiveContainer width="100%" height={160}>
                      <AreaChart
                        data={[
                          {
                            month: "Last Month",
                            balance:
                              visualizations.balance_analysis.opening_balance
                                .last_month,
                          },
                          {
                            month: "Current Month",
                            balance:
                              visualizations.balance_analysis.opening_balance
                                .current_month,
                          },
                        ]}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          strokeOpacity={0.4}
                        />
                        <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                          type="monotone"
                          dataKey="balance"
                          stroke="#3b82f6"
                          fill="#3b82f6"
                          fillOpacity={0.6}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </GlassCard>

                  <GlassCard className="p-4">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3">
                      Closing Balance Trend
                    </h3>
                    <ResponsiveContainer width="100%" height={160}>
                      <AreaChart
                        data={[
                          {
                            month: "Last Month",
                            balance:
                              visualizations.balance_analysis.closing_balance
                                .last_month,
                          },
                          {
                            month: "Current Month",
                            balance:
                              visualizations.balance_analysis.closing_balance
                                .current_month,
                          },
                        ]}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid
                          strokeDasharray="3 3"
                          strokeOpacity={0.4}
                        />
                        <XAxis dataKey="month" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 10 }} />
                        <Tooltip content={<CustomTooltip />} />
                        <Area
                          type="monotone"
                          dataKey="balance"
                          stroke="#10b981"
                          fill="#10b981"
                          fillOpacity={0.6}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </GlassCard>
                </div>
              </>
            ) : (
              <GlassCard className="p-5 text-center">
                <Loader2 className="w-5 h-5 text-blue-600 mx-auto mb-2 animate-spin" />
                <p className="text-gray-600 text-xs">
                  Loading visualization data...
                </p>
              </GlassCard>
            )}
          </div>
        )}

        {/* Neural Network Tab */}
        {activeTab === "neural-network" && (
          <div className="space-y-4">
            {!neuralNetworkResults ? (
              <GlassCard className="p-5 text-center">
                <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-3 shadow-md">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-base font-semibold text-gray-900 mb-1.5">
                  AI-Powered Analysis
                </h3>
                <p className="text-gray-600 mb-4 text-xs">
                  Use neural network to analyze transaction patterns and predict
                  high-value transactions
                </p>
                <button
                  onClick={runNeuralNetworkAnalysis}
                  disabled={loadingNN}
                  className="px-4 py-2 bg-gradient-to-r from-purple-600 to-purple-700 text-white rounded-lg hover:from-purple-700 hover:to-purple-800 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center shadow-md hover:shadow-lg transform hover:scale-[1.02] text-xs"
                >
                  {loadingNN ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                      Training Neural Network...
                    </>
                  ) : (
                    <>
                      <Brain className="w-3.5 h-3.5 mr-1.5" />
                      Run AI Analysis
                    </>
                  )}
                </button>
              </GlassCard>
            ) : (
              <>
                {/* Model Performance */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                  <GlassCard className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-gray-900">
                        Model Accuracy
                      </h3>
                      <div className="p-1.5 rounded-lg bg-green-100/70">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                    </div>
                    <div className="space-y-2.5">
                      <div>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="text-gray-600">
                            Training Accuracy
                          </span>
                          <span className="font-medium">
                            {
                              neuralNetworkResults.model_performance
                                .train_accuracy
                            }
                            %
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1">
                          <div
                            className="bg-blue-500 h-1 rounded-full transition-all duration-500"
                            style={{
                              width: `${neuralNetworkResults.model_performance.train_accuracy}%`,
                            }}
                          ></div>
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-[10px] mb-1">
                          <span className="text-gray-600">Test Accuracy</span>
                          <span className="font-medium">
                            {
                              neuralNetworkResults.model_performance
                                .test_accuracy
                            }
                            %
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1">
                          <div
                            className="bg-green-500 h-1 rounded-full transition-all duration-500"
                            style={{
                              width: `${neuralNetworkResults.model_performance.test_accuracy}%`,
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </GlassCard>

                  <GlassCard className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-gray-900">
                        Dataset Info
                      </h3>
                      <div className="p-1.5 rounded-lg bg-blue-100/70">
                        <FileText className="w-4 h-4 text-blue-500" />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600 text-xs">
                          Total Samples
                        </span>
                        <span className="font-medium text-xs">
                          {formatNumber(
                            neuralNetworkResults.model_performance.total_samples
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 text-xs">
                          Training Samples
                        </span>
                        <span className="font-medium text-xs">
                          {formatNumber(
                            neuralNetworkResults.model_performance
                              .training_samples
                          )}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600 text-xs">
                          Test Samples
                        </span>
                        <span className="font-medium text-xs">
                          {formatNumber(
                            neuralNetworkResults.model_performance.test_samples
                          )}
                        </span>
                      </div>
                    </div>
                  </GlassCard>

                  <GlassCard className="p-4">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-sm font-semibold text-gray-900">
                        Trend Analysis
                      </h3>
                      <div
                        className={`p-1.5 rounded-lg ${
                          neuralNetworkResults.monthly_insights.trend ===
                          "Increasing"
                            ? "bg-green-100/70"
                            : "bg-red-100/70"
                        }`}
                      >
                        {neuralNetworkResults.monthly_insights.trend ===
                        "Increasing" ? (
                          <TrendingUp className="w-4 h-4 text-green-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-center">
                        <p className="text-lg font-bold text-gray-900">
                          {neuralNetworkResults.monthly_insights.trend}
                        </p>
                        <p className="text-gray-600 mt-1 text-xs">
                          High-value transaction probability
                        </p>
                      </div>
                    </div>
                  </GlassCard>
                </div>

                {/* Monthly Insights */}
                <GlassCard className="p-4">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">
                    High-Value Transaction Probability by Month
                  </h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-gray-50/70 rounded-lg backdrop-blur-sm">
                      <p className="text-[10px] text-gray-600 mb-1.5">
                        Last Month
                      </p>
                      <p className="text-2xl font-bold text-gray-900">
                        {(
                          neuralNetworkResults.monthly_insights
                            .last_month_high_value_probability * 100
                        ).toFixed(2)}
                        %
                      </p>
                      <p className="text-[10px] text-gray-500 mt-1.5">
                        Probability of high-value transactions
                      </p>
                    </div>
                    <div className="text-center p-4 bg-blue-50/70 rounded-lg backdrop-blur-sm">
                      <p className="text-[10px] text-gray-600 mb-1.5">
                        Current Month
                      </p>
                      <p className="text-2xl font-bold text-blue-600">
                        {(
                          neuralNetworkResults.monthly_insights
                            .current_month_high_value_probability * 100
                        ).toFixed(2)}
                        %
                      </p>
                      <p className="text-[10px] text-gray-500 mt-1.5">
                        Probability of high-value transactions
                      </p>
                    </div>
                  </div>
                </GlassCard>

                {/* Training Loss Chart */}
                <GlassCard className="p-4">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">
                    Neural Network Training Progress
                  </h3>
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart
                      data={neuralNetworkResults.training_loss.map(
                        (loss, index) => ({
                          epoch: index * 10,
                          loss: loss,
                        })
                      )}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        strokeOpacity={0.4}
                      />
                      <XAxis
                        dataKey="epoch"
                        label={{
                          value: "Epoch",
                          position: "insideBottom",
                          offset: -5,
                          fontSize: 10,
                        }}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis
                        label={{
                          value: "Loss",
                          angle: -90,
                          position: "insideLeft",
                          fontSize: 10,
                        }}
                        tick={{ fontSize: 10 }}
                      />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                        name="Training Loss"
                        activeDot={{ r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-[10px] text-gray-600 mt-2.5">
                    The training loss decreases over epochs, indicating the
                    neural network is learning patterns in the data effectively.
                  </p>
                </GlassCard>
              </>
            )}
          </div>
        )}

        {/* Bank Statement Tab */}
        {activeTab === "bank-statement" && <BankStatementView />}
      </div>
    </div>
  );
};

// Add custom animation keyframes to your styles
const style = document.createElement("style");
style.textContent = `
  @keyframes spin-slow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
  
  @keyframes spin-reverse {
    from { transform: rotate(0deg); }
    to { transform: rotate(-360deg); }
  }
  
  @keyframes float {
    0% { transform: translateY(0) translateX(0); opacity: 0; }
    50% { opacity: 0.8; }
    100% { transform: translateY(-100px) translateX(30px); opacity: 0; }
  }
  
  @keyframes loading-bar {
    0% { width: 0%; }
    50% { width: 100%; }
    100% { width: 0%; }
  }
  
  .animate-spin-slow {
    animation: spin-slow 6s linear infinite;
  }
  
  .animate-spin-reverse {
    animation: spin-reverse 4s linear infinite;
  }
  
  .animate-float {
    animation: float 15s ease-in infinite;
  }
  
  .animate-loading-bar {
    animation: loading-bar 2s ease-in-out infinite;
  }
`;
document.head.appendChild(style);

export default EnhancedBankingDashboard;
