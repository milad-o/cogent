"""Classification functions for SSIS executables and components."""

from __future__ import annotations


def classify_executable(exe_type: str) -> str:
    """Classify an executable into an entity type based on CreationName."""
    exe_type_lower = exe_type.lower()

    # Containers
    if "sequence" in exe_type_lower:
        return "SequenceContainer"
    elif "forloop" in exe_type_lower:
        return "ForLoopContainer"
    elif "foreachloop" in exe_type_lower:
        return "ForEachLoopContainer"

    # Data Flow
    elif "pipeline" in exe_type_lower or "dataflow" in exe_type_lower:
        return "DataFlowTask"

    # Package/Process execution
    elif "executepackage" in exe_type_lower:
        return "ExecutePackageTask"
    elif "executeprocess" in exe_type_lower:
        return "ExecuteProcessTask"
    elif "executedts" in exe_type_lower:
        return "ExecuteDTSTask"

    # SQL/Database
    elif "sqltask" in exe_type_lower or "executesql" in exe_type_lower:
        return "SQLTask"
    elif "bulkinsert" in exe_type_lower:
        return "BulkInsertTask"
    elif "transferdatabase" in exe_type_lower:
        return "TransferDatabaseTask"
    elif "transferjobs" in exe_type_lower:
        return "TransferJobsTask"
    elif "transferlogins" in exe_type_lower:
        return "TransferLoginsTask"
    elif "transferobjects" in exe_type_lower:
        return "TransferObjectsTask"
    elif "transfererror" in exe_type_lower:
        return "TransferErrorMessagesTask"
    elif "transferstoredprocedures" in exe_type_lower:
        return "TransferStoredProceduresTask"

    # Scripting
    elif "scripttask" in exe_type_lower:
        return "ScriptTask"
    elif "activexscript" in exe_type_lower:
        return "ActiveXScriptTask"

    # File operations
    elif "filetask" in exe_type_lower or "filesystem" in exe_type_lower:
        return "FileSystemTask"
    elif "ftptask" in exe_type_lower:
        return "FTPTask"

    # Communication
    elif "sendemail" in exe_type_lower or "sendmail" in exe_type_lower:
        return "SendMailTask"
    elif "messagequeue" in exe_type_lower or "msmq" in exe_type_lower:
        return "MessageQueueTask"
    elif "webservice" in exe_type_lower:
        return "WebServiceTask"

    # XML
    elif "xmltask" in exe_type_lower:
        return "XMLTask"

    # WMI
    elif "wmidatareader" in exe_type_lower:
        return "WMIDataReaderTask"
    elif "wmieventwatcher" in exe_type_lower:
        return "WMIEventWatcherTask"

    # Analysis Services
    elif "asprocessing" in exe_type_lower or "analysisservices" in exe_type_lower:
        return "AnalysisServicesTask"
    elif "asdatadef" in exe_type_lower:
        return "ASDataDefinitionTask"

    # Data mining
    elif "datamining" in exe_type_lower:
        return "DataMiningTask"

    # Expression
    elif "expressiontask" in exe_type_lower:
        return "ExpressionTask"

    # Maintenance
    elif "maintenance" in exe_type_lower:
        return "MaintenanceTask"
    elif "backup" in exe_type_lower:
        return "BackupTask"
    elif "shrink" in exe_type_lower:
        return "ShrinkDatabaseTask"
    elif "rebuild" in exe_type_lower or "reorganize" in exe_type_lower:
        return "IndexTask"
    elif "updatestatistics" in exe_type_lower:
        return "UpdateStatisticsTask"
    elif "checkintegrity" in exe_type_lower:
        return "CheckIntegrityTask"
    elif "cleanuphistory" in exe_type_lower:
        return "CleanupHistoryTask"

    # CDC (Change Data Capture)
    elif "cdccontrol" in exe_type_lower:
        return "CDCControlTask"

    # Hadoop/Big Data
    elif "hadoop" in exe_type_lower:
        return "HadoopTask"
    elif "hive" in exe_type_lower:
        return "HiveTask"
    elif "pig" in exe_type_lower:
        return "PigTask"

    # Azure
    elif "azure" in exe_type_lower:
        return "AzureTask"

    else:
        return "Task"


def classify_component(comp_type: str, contact_info: str) -> str:
    """Classify a data flow component into an entity type."""
    combined = (comp_type + contact_info).lower()

    if "source" in combined:
        return "Source"
    elif "destination" in combined:
        return "Destination"
    elif "lookup" in combined:
        return "Lookup"
    elif "merge" in combined:
        return "MergeTransform"
    elif "conditional" in combined or "split" in combined:
        return "ConditionalSplit"
    elif "derived" in combined:
        return "DerivedColumn"
    elif "aggregate" in combined:
        return "Aggregate"
    elif "sort" in combined:
        return "Sort"
    elif "union" in combined:
        return "UnionAll"
    elif "multicast" in combined:
        return "Multicast"
    elif "rowcount" in combined:
        return "RowCount"
    elif "convert" in combined:
        return "DataConversion"
    elif "script" in combined:
        return "ScriptComponent"
    else:
        return "Transform"
