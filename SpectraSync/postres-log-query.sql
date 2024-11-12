-- Select columns from the logs table for relevant log details
SELECT log_time,           -- Timestamp of the log entry
       log_level,          -- Severity level of the log entry (e.g., ERROR, CRITICAL)
       message,            -- Log message text
       module,             -- Name of the module where the log was generated
       func_name,          -- Function name where the log was generated
       line_no             -- Line number in the code where the log was generated
FROM logs
WHERE log_level IN ('WARNING','ERROR')  -- Filter for ERROR and CRITICAL logs only
ORDER BY log_time desc;    -- Order results by most recent log entries
