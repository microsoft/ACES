CREATE USER 'admin'@'%' IDENTIFIED BY 'admin';

GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%';

FLUSH PRIVILEGES;

CREATE DATABASE IF NOT EXISTS env_monitor_db;

USE env_monitor_db;

CREATE TABLE AADManagedIdentitySignInLogs (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CorrelationId TEXT,
    ResourceGroup TEXT,
    Identity TEXT,
    Level TEXT,
    Location TEXT,
    AppId TEXT,
    AuthenticationContextClassReferences TEXT,
    AuthenticationProcessingDetails TEXT,
    ConditionalAccessPolicies TEXT,
    ConditionalAccessPoliciesV2 TEXT,
    ConditionalAccessStatus TEXT,
    FederatedCredentialId TEXT,
    Id TEXT,
    IPAddress TEXT,
    LocationDetails TEXT,
    ResourceDisplayName TEXT,
    ResourceIdentity TEXT,
    ResourceServicePrincipalId TEXT,
    ServicePrincipalCredentialKeyId TEXT,
    ServicePrincipalCredentialThumbprint TEXT,
    ServicePrincipalId TEXT,
    ServicePrincipalName TEXT,
    UniqueTokenIdentifier TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AADManagedIdentitySignInLogs.csv'
INTO TABLE AADManagedIdentitySignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AADNonInteractiveUserSignInLogs (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CorrelationId TEXT,
    ResourceGroup TEXT,
    Identity TEXT,
    Level TEXT,
    Location TEXT,
    AlternateSignInName TEXT,
    AppDisplayName TEXT,
    AppId TEXT,
    AppliedEventListeners TEXT,
    AuthenticationContextClassReferences TEXT,
    AuthenticationDetails TEXT,
    AuthenticationMethodsUsed TEXT,
    AuthenticationProcessingDetails TEXT,
    AuthenticationProtocol TEXT,
    AuthenticationRequirement TEXT,
    AuthenticationRequirementPolicies TEXT,
    AutonomousSystemNumber TEXT,
    ClientAppUsed TEXT,
    ConditionalAccessPolicies TEXT,
    ConditionalAccessPoliciesV2 TEXT,
    ConditionalAccessStatus TEXT,
    CreatedDateTime TEXT,
    CrossTenantAccessType TEXT,
    DeviceDetail TEXT,
    HomeTenantId TEXT,
    Id TEXT,
    IPAddress TEXT,
    IsInteractive TEXT,
    IsRisky TEXT,
    LocationDetails TEXT,
    MfaDetail TEXT,
    NetworkLocationDetails TEXT,
    OriginalRequestId TEXT,
    ProcessingTimeInMs TEXT,
    ResourceDisplayName TEXT,
    ResourceIdentity TEXT,
    ResourceServicePrincipalId TEXT,
    ResourceTenantId TEXT,
    RiskDetail TEXT,
    RiskEventTypes TEXT,
    RiskEventTypes_V2 TEXT,
    RiskLevelAggregated TEXT,
    RiskLevelDuringSignIn TEXT,
    RiskState TEXT,
    ServicePrincipalId TEXT,
    SessionLifetimePolicies TEXT,
    SignInEventTypes TEXT,
    SignInIdentifierType TEXT,
    Status TEXT,
    TokenIssuerName TEXT,
    TokenIssuerType TEXT,
    UniqueTokenIdentifier TEXT,
    UserAgent TEXT,
    UserDisplayName TEXT,
    UserId TEXT,
    UserPrincipalName TEXT,
    UserType TEXT,
    Type TEXT,
    rn TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AADNonInteractiveUserSignInLogs/._AADNonInteractiveUserSignInLogs_0.csv'
INTO TABLE AADNonInteractiveUserSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



LOAD DATA INFILE '/var/lib/mysql-files/AADNonInteractiveUserSignInLogs/._AADNonInteractiveUserSignInLogs_1.csv'
INTO TABLE AADNonInteractiveUserSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



LOAD DATA INFILE '/var/lib/mysql-files/AADNonInteractiveUserSignInLogs/._AADNonInteractiveUserSignInLogs_2.csv'
INTO TABLE AADNonInteractiveUserSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



LOAD DATA INFILE '/var/lib/mysql-files/AADNonInteractiveUserSignInLogs/AADNonInteractiveUserSignInLogs_0.csv'
INTO TABLE AADNonInteractiveUserSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



LOAD DATA INFILE '/var/lib/mysql-files/AADNonInteractiveUserSignInLogs/AADNonInteractiveUserSignInLogs_1.csv'
INTO TABLE AADNonInteractiveUserSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;



LOAD DATA INFILE '/var/lib/mysql-files/AADNonInteractiveUserSignInLogs/AADNonInteractiveUserSignInLogs_2.csv'
INTO TABLE AADNonInteractiveUserSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AADProvisioningLogs (
    TenantId TEXT,
    AADTenantId TEXT,
    TimeGenerated TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CorrelationId TEXT,
    Action TEXT,
    ChangeId TEXT,
    CycleId TEXT,
    Id TEXT,
    InitiatedBy TEXT,
    JobId TEXT,
    ModifiedProperties TEXT,
    ProvisioningSteps TEXT,
    ServicePrincipal TEXT,
    SourceIdentity TEXT,
    SourceSystem TEXT,
    StatusInfo TEXT,
    TargetIdentity TEXT,
    TargetSystem TEXT,
    ProvisioningAction TEXT,
    ProvisioningStatusInfo TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AADProvisioningLogs.csv'
INTO TABLE AADProvisioningLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AADServicePrincipalSignInLogs (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CorrelationId TEXT,
    ResourceGroup TEXT,
    Identity TEXT,
    Level TEXT,
    Location TEXT,
    AppId TEXT,
    AuthenticationContextClassReferences TEXT,
    AuthenticationProcessingDetails TEXT,
    ConditionalAccessPolicies TEXT,
    ConditionalAccessPoliciesV2 TEXT,
    ConditionalAccessStatus TEXT,
    FederatedCredentialId TEXT,
    Id TEXT,
    IPAddress TEXT,
    LocationDetails TEXT,
    ResourceDisplayName TEXT,
    ResourceIdentity TEXT,
    ResourceServicePrincipalId TEXT,
    ServicePrincipalCredentialKeyId TEXT,
    ServicePrincipalCredentialThumbprint TEXT,
    ServicePrincipalId TEXT,
    ServicePrincipalName TEXT,
    UniqueTokenIdentifier TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AADServicePrincipalSignInLogs.csv'
INTO TABLE AADServicePrincipalSignInLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AlertEvidence (
    TenantId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    AlertId TEXT,
    Title TEXT,
    Categories TEXT,
    AttackTechniques TEXT,
    ServiceSource TEXT,
    DetectionSource TEXT,
    EntityType TEXT,
    EvidenceRole TEXT,
    EvidenceDirection TEXT,
    FileName TEXT,
    FolderPath TEXT,
    SHA1 TEXT,
    SHA256 TEXT,
    FileSize TEXT,
    ThreatFamily TEXT,
    RemoteIP TEXT,
    RemoteUrl TEXT,
    AccountName TEXT,
    AccountDomain TEXT,
    AccountSid TEXT,
    AccountObjectId TEXT,
    AccountUpn TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    LocalIP TEXT,
    NetworkMessageId TEXT,
    EmailSubject TEXT,
    ApplicationId TEXT,
    Application TEXT,
    OAuthApplicationId TEXT,
    ProcessCommandLine TEXT,
    AdditionalFields TEXT,
    RegistryKey TEXT,
    RegistryValueName TEXT,
    RegistryValueData TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AlertEvidence.csv'
INTO TABLE AlertEvidence
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AlertInfo (
    TenantId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    AlertId TEXT,
    Title TEXT,
    Category TEXT,
    Severity TEXT,
    ServiceSource TEXT,
    DetectionSource TEXT,
    AttackTechniques TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AlertInfo.csv'
INTO TABLE AlertInfo
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AuditLogs (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    ResourceId TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CorrelationId TEXT,
    Resource TEXT,
    ResourceGroup TEXT,
    ResourceProvider TEXT,
    Identity TEXT,
    Level TEXT,
    Location TEXT,
    AdditionalDetails TEXT,
    Id TEXT,
    InitiatedBy TEXT,
    LoggedByService TEXT,
    Result TEXT,
    ResultReason TEXT,
    TargetResources TEXT,
    AADTenantId TEXT,
    ActivityDisplayName TEXT,
    ActivityDateTime TEXT,
    AADOperationType TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AuditLogs.csv'
INTO TABLE AuditLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AZFWApplicationRule (
    TenantId TEXT,
    TimeGenerated TEXT,
    Protocol TEXT,
    SourceIp TEXT,
    SourcePort TEXT,
    DestinationPort TEXT,
    Fqdn TEXT,
    TargetUrl TEXT,
    Action TEXT,
    Policy TEXT,
    RuleCollectionGroup TEXT,
    RuleCollection TEXT,
    Rule TEXT,
    ActionReason TEXT,
    IsTlsInspected TEXT,
    WebCategory TEXT,
    IsExplicitProxyRequest TEXT,
    SourceSystem TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AZFWApplicationRule.csv'
INTO TABLE AZFWApplicationRule
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AZFWApplicationRuleAggregation (
    TenantId TEXT,
    TimeGenerated TEXT,
    Protocol TEXT,
    SourceIp TEXT,
    DestinationPort TEXT,
    Fqdn TEXT,
    TargetUrl TEXT,
    Action TEXT,
    Policy TEXT,
    RuleCollectionGroup TEXT,
    RuleCollection TEXT,
    Rule TEXT,
    ActionReason TEXT,
    ApplicationRuleCount TEXT,
    SourceSystem TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AZFWApplicationRuleAggregation.csv'
INTO TABLE AZFWApplicationRuleAggregation
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AZFWDnsQuery (
    TenantId TEXT,
    TimeGenerated TEXT,
    SourceIp TEXT,
    SourcePort TEXT,
    QueryId TEXT,
    QueryType TEXT,
    QueryClass TEXT,
    QueryName TEXT,
    Protocol TEXT,
    RequestSize TEXT,
    DnssecOkBit TEXT,
    EDNS0BufferSize TEXT,
    ResponseCode TEXT,
    ResponseFlags TEXT,
    ResponseSize TEXT,
    RequestDurationSecs TEXT,
    ErrorNumber TEXT,
    ErrorMessage TEXT,
    SourceSystem TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AZFWDnsQuery.csv'
INTO TABLE AZFWDnsQuery
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AZFWNetworkRule (
    TenantId TEXT,
    TimeGenerated TEXT,
    Protocol TEXT,
    SourceIp TEXT,
    SourcePort TEXT,
    DestinationIp TEXT,
    DestinationPort TEXT,
    Action TEXT,
    Policy TEXT,
    RuleCollectionGroup TEXT,
    RuleCollection TEXT,
    Rule TEXT,
    ActionReason TEXT,
    SourceSystem TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AZFWNetworkRule.csv'
INTO TABLE AZFWNetworkRule
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AZFWNetworkRuleAggregation (
    TenantId TEXT,
    TimeGenerated TEXT,
    Protocol TEXT,
    SourceIp TEXT,
    DestinationIp TEXT,
    DestinationPort TEXT,
    Action TEXT,
    ActionReason TEXT,
    Policy TEXT,
    RuleCollectionGroup TEXT,
    RuleCollection TEXT,
    Rule TEXT,
    IsDefaultRule TEXT,
    NetworkRuleCount TEXT,
    SourceSystem TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AZFWNetworkRuleAggregation.csv'
INTO TABLE AZFWNetworkRuleAggregation
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE AzureMetrics (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    ResourceId TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CallerIpAddress TEXT,
    CorrelationId TEXT,
    Resource TEXT,
    ResourceGroup TEXT,
    ResourceProvider TEXT,
    SubscriptionId TEXT,
    MetricName TEXT,
    Total TEXT,
    Count TEXT,
    Maximum TEXT,
    Minimum TEXT,
    Average TEXT,
    TimeGrain TEXT,
    UnitName TEXT,
    RemoteIPCountry TEXT,
    RemoteIPLatitude TEXT,
    RemoteIPLongitude TEXT,
    MaliciousIP TEXT,
    IndicatorThreatType TEXT,
    Description TEXT,
    TLPLevel TEXT,
    Confidence TEXT,
    Severity TEXT,
    FirstReportedDateTime TEXT,
    LastReportedDateTime TEXT,
    IsActive TEXT,
    ReportReferenceLink TEXT,
    AdditionalInformation TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/AzureMetrics.csv'
INTO TABLE AzureMetrics
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE CloudAppEvents (
    TenantId TEXT,
    AccountId TEXT,
    AccountType TEXT,
    AdditionalFields TEXT,
    RawEventData TEXT,
    ReportId TEXT,
    ObjectId TEXT,
    ObjectType TEXT,
    ObjectName TEXT,
    ActivityObjects TEXT,
    ActivityType TEXT,
    UserAgent TEXT,
    ISP TEXT,
    City TEXT,
    CountryCode TEXT,
    IsAnonymousProxy TEXT,
    IsExternalUser TEXT,
    IsImpersonated TEXT,
    IPAddress TEXT,
    IPCategory TEXT,
    IPTags TEXT,
    OSPlatform TEXT,
    DeviceType TEXT,
    IsAdminOperation TEXT,
    AccountDisplayName TEXT,
    AccountObjectId TEXT,
    AppInstanceId TEXT,
    ApplicationId TEXT,
    Application TEXT,
    ActionType TEXT,
    UserAgentTags TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/CloudAppEvents.csv'
INTO TABLE CloudAppEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceEvents (
    TenantId TEXT,
    AccountDomain TEXT,
    AccountName TEXT,
    AccountSid TEXT,
    ActionType TEXT,
    AdditionalFields TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    FileName TEXT,
    FileOriginIP TEXT,
    FileOriginUrl TEXT,
    FolderPath TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessLogonId TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    LocalIP TEXT,
    LocalPort TEXT,
    LogonId TEXT,
    MD5 TEXT,
    MachineGroup TEXT,
    ProcessCommandLine TEXT,
    ProcessId TEXT,
    ProcessTokenElevation TEXT,
    RegistryKey TEXT,
    RegistryValueData TEXT,
    RegistryValueName TEXT,
    RemoteDeviceName TEXT,
    RemoteIP TEXT,
    RemotePort TEXT,
    RemoteUrl TEXT,
    ReportId TEXT,
    SHA1 TEXT,
    SHA256 TEXT,
    Timestamp TEXT,
    TimeGenerated TEXT,
    FileSize TEXT,
    InitiatingProcessCreationTime TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    ProcessCreationTime TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceEvents.csv'
INTO TABLE DeviceEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceFileCertificateInfo (
    TenantId TEXT,
    CertificateSerialNumber TEXT,
    CrlDistributionPointUrls TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    IsRootSignerMicrosoft TEXT,
    IsSigned TEXT,
    IsTrusted TEXT,
    Issuer TEXT,
    IssuerHash TEXT,
    MachineGroup TEXT,
    ReportId TEXT,
    SHA1 TEXT,
    SignatureType TEXT,
    Signer TEXT,
    SignerHash TEXT,
    Timestamp TEXT,
    TimeGenerated TEXT,
    CertificateCountersignatureTime TEXT,
    CertificateCreationTime TEXT,
    CertificateExpirationTime TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceFileCertificateInfo.csv'
INTO TABLE DeviceFileCertificateInfo
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceFileEvents (
    TenantId TEXT,
    ActionType TEXT,
    AdditionalFields TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    FileName TEXT,
    FileOriginIP TEXT,
    FileOriginReferrerUrl TEXT,
    FileOriginUrl TEXT,
    FileSize TEXT,
    FolderPath TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessIntegrityLevel TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    InitiatingProcessTokenElevation TEXT,
    IsAzureInfoProtectionApplied TEXT,
    MD5 TEXT,
    MachineGroup TEXT,
    PreviousFileName TEXT,
    PreviousFolderPath TEXT,
    ReportId TEXT,
    RequestAccountDomain TEXT,
    RequestAccountName TEXT,
    RequestAccountSid TEXT,
    RequestProtocol TEXT,
    RequestSourceIP TEXT,
    RequestSourcePort TEXT,
    SHA1 TEXT,
    SHA256 TEXT,
    SensitivityLabel TEXT,
    SensitivitySubLabel TEXT,
    ShareName TEXT,
    Timestamp TEXT,
    TimeGenerated TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessCreationTime TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceFileEvents.csv'
INTO TABLE DeviceFileEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceImageLoadEvents (
    TenantId TEXT,
    ActionType TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    FileName TEXT,
    FolderPath TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessIntegrityLevel TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    InitiatingProcessTokenElevation TEXT,
    MD5 TEXT,
    MachineGroup TEXT,
    ReportId TEXT,
    SHA1 TEXT,
    SHA256 TEXT,
    Timestamp TEXT,
    TimeGenerated TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessCreationTime TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    FileSize TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceImageLoadEvents.csv'
INTO TABLE DeviceImageLoadEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceInfo (
    TenantId TEXT,
    AdditionalFields TEXT,
    ClientVersion TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    DeviceObjectId TEXT,
    IsAzureADJoined TEXT,
    LoggedOnUsers TEXT,
    MachineGroup TEXT,
    OSArchitecture TEXT,
    OSBuild TEXT,
    OSPlatform TEXT,
    OSVersion TEXT,
    PublicIP TEXT,
    RegistryDeviceTag TEXT,
    ReportId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    AadDeviceId TEXT,
    DeviceCategory TEXT,
    DeviceSubtype TEXT,
    DeviceType TEXT,
    JoinType TEXT,
    MergedDeviceIds TEXT,
    MergedToDeviceId TEXT,
    Model TEXT,
    OnboardingStatus TEXT,
    OSDistribution TEXT,
    OSVersionInfo TEXT,
    Vendor TEXT,
    SensorHealthState TEXT,
    IsExcluded TEXT,
    ExclusionReason TEXT,
    AssetValue TEXT,
    ExposureLevel TEXT,
    IsInternetFacing TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceInfo.csv'
INTO TABLE DeviceInfo
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceLogonEvents (
    TenantId TEXT,
    AccountDomain TEXT,
    AccountName TEXT,
    AccountSid TEXT,
    ActionType TEXT,
    AdditionalFields TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    FailureReason TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessIntegrityLevel TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    InitiatingProcessTokenElevation TEXT,
    IsLocalAdmin TEXT,
    LogonId TEXT,
    LogonType TEXT,
    MachineGroup TEXT,
    Protocol TEXT,
    RemoteDeviceName TEXT,
    RemoteIP TEXT,
    RemoteIPType TEXT,
    RemotePort TEXT,
    ReportId TEXT,
    Timestamp TEXT,
    TimeGenerated TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessCreationTime TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceLogonEvents.csv'
INTO TABLE DeviceLogonEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceNetworkEvents (
    TenantId TEXT,
    ActionType TEXT,
    AdditionalFields TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessIntegrityLevel TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    InitiatingProcessTokenElevation TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    LocalIP TEXT,
    LocalIPType TEXT,
    LocalPort TEXT,
    MachineGroup TEXT,
    Protocol TEXT,
    RemoteIP TEXT,
    RemoteIPType TEXT,
    RemotePort TEXT,
    RemoteUrl TEXT,
    ReportId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessCreationTime TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceNetworkEvents.csv'
INTO TABLE DeviceNetworkEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceNetworkInfo (
    TenantId TEXT,
    ConnectedNetworks TEXT,
    DefaultGateways TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    DnsAddresses TEXT,
    IPAddresses TEXT,
    IPv4Dhcp TEXT,
    IPv6Dhcp TEXT,
    MacAddress TEXT,
    MachineGroup TEXT,
    NetworkAdapterName TEXT,
    NetworkAdapterStatus TEXT,
    NetworkAdapterType TEXT,
    ReportId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    TunnelType TEXT,
    NetworkAdapterVendor TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceNetworkInfo.csv'
INTO TABLE DeviceNetworkInfo
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceProcessEvents (
    TenantId TEXT,
    AccountDomain TEXT,
    AccountName TEXT,
    AccountObjectId TEXT,
    AccountSid TEXT,
    AccountUpn TEXT,
    ActionType TEXT,
    AdditionalFields TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    FileName TEXT,
    FolderPath TEXT,
    FileSize TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessIntegrityLevel TEXT,
    InitiatingProcessLogonId TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    InitiatingProcessTokenElevation TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    LogonId TEXT,
    MD5 TEXT,
    MachineGroup TEXT,
    ProcessCommandLine TEXT,
    ProcessCreationTime TEXT,
    ProcessId TEXT,
    ProcessIntegrityLevel TEXT,
    ProcessTokenElevation TEXT,
    ProcessVersionInfoCompanyName TEXT,
    ProcessVersionInfoProductName TEXT,
    ProcessVersionInfoProductVersion TEXT,
    ProcessVersionInfoInternalFileName TEXT,
    ProcessVersionInfoOriginalFileName TEXT,
    ProcessVersionInfoFileDescription TEXT,
    InitiatingProcessSignerType TEXT,
    InitiatingProcessSignatureStatus TEXT,
    ReportId TEXT,
    SHA1 TEXT,
    SHA256 TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessCreationTime TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceProcessEvents.csv'
INTO TABLE DeviceProcessEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE DeviceRegistryEvents (
    TenantId TEXT,
    ActionType TEXT,
    AppGuardContainerId TEXT,
    DeviceId TEXT,
    DeviceName TEXT,
    InitiatingProcessAccountDomain TEXT,
    InitiatingProcessAccountName TEXT,
    InitiatingProcessAccountObjectId TEXT,
    InitiatingProcessAccountSid TEXT,
    InitiatingProcessAccountUpn TEXT,
    InitiatingProcessCommandLine TEXT,
    InitiatingProcessFileName TEXT,
    InitiatingProcessFolderPath TEXT,
    InitiatingProcessId TEXT,
    InitiatingProcessIntegrityLevel TEXT,
    InitiatingProcessMD5 TEXT,
    InitiatingProcessParentFileName TEXT,
    InitiatingProcessParentId TEXT,
    InitiatingProcessSHA1 TEXT,
    InitiatingProcessSHA256 TEXT,
    InitiatingProcessTokenElevation TEXT,
    InitiatingProcessFileSize TEXT,
    InitiatingProcessVersionInfoCompanyName TEXT,
    InitiatingProcessVersionInfoProductName TEXT,
    InitiatingProcessVersionInfoProductVersion TEXT,
    InitiatingProcessVersionInfoInternalFileName TEXT,
    InitiatingProcessVersionInfoOriginalFileName TEXT,
    InitiatingProcessVersionInfoFileDescription TEXT,
    MachineGroup TEXT,
    PreviousRegistryKey TEXT,
    PreviousRegistryValueData TEXT,
    PreviousRegistryValueName TEXT,
    RegistryKey TEXT,
    RegistryValueData TEXT,
    RegistryValueName TEXT,
    RegistryValueType TEXT,
    ReportId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    InitiatingProcessParentCreationTime TEXT,
    InitiatingProcessCreationTime TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/DeviceRegistryEvents.csv'
INTO TABLE DeviceRegistryEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE EmailAttachmentInfo (
    TenantId TEXT,
    FileName TEXT,
    FileType TEXT,
    NetworkMessageId TEXT,
    RecipientEmailAddress TEXT,
    RecipientObjectId TEXT,
    ReportId TEXT,
    SHA256 TEXT,
    SenderDisplayName TEXT,
    SenderObjectId TEXT,
    ThreatTypes TEXT,
    SenderFromAddress TEXT,
    ThreatNames TEXT,
    DetectionMethods TEXT,
    FileSize TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/EmailAttachmentInfo.csv'
INTO TABLE EmailAttachmentInfo
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE EmailEvents (
    TenantId TEXT,
    AttachmentCount TEXT,
    AuthenticationDetails TEXT,
    AdditionalFields TEXT,
    ConfidenceLevel TEXT,
    Connectors TEXT,
    DetectionMethods TEXT,
    DeliveryAction TEXT,
    DeliveryLocation TEXT,
    EmailClusterId TEXT,
    EmailDirection TEXT,
    EmailLanguage TEXT,
    EmailAction TEXT,
    EmailActionPolicy TEXT,
    EmailActionPolicyGuid TEXT,
    OrgLevelAction TEXT,
    OrgLevelPolicy TEXT,
    InternetMessageId TEXT,
    NetworkMessageId TEXT,
    RecipientEmailAddress TEXT,
    RecipientObjectId TEXT,
    ReportId TEXT,
    SenderDisplayName TEXT,
    SenderFromAddress TEXT,
    SenderFromDomain TEXT,
    SenderObjectId TEXT,
    SenderIPv4 TEXT,
    SenderIPv6 TEXT,
    SenderMailFromAddress TEXT,
    SenderMailFromDomain TEXT,
    Subject TEXT,
    ThreatTypes TEXT,
    ThreatNames TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    UrlCount TEXT,
    UserLevelAction TEXT,
    UserLevelPolicy TEXT,
    BulkComplaintLevel TEXT,
    LatestDeliveryLocation TEXT,
    LatestDeliveryAction TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/EmailEvents.csv'
INTO TABLE EmailEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE EmailPostDeliveryEvents (
    TenantId TEXT,
    Action TEXT,
    ActionResult TEXT,
    ActionTrigger TEXT,
    ActionType TEXT,
    DeliveryLocation TEXT,
    InternetMessageId TEXT,
    NetworkMessageId TEXT,
    RecipientEmailAddress TEXT,
    ReportId TEXT,
    ThreatTypes TEXT,
    DetectionMethods TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/EmailPostDeliveryEvents.csv'
INTO TABLE EmailPostDeliveryEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE EmailUrlInfo (
    TenantId TEXT,
    NetworkMessageId TEXT,
    ReportId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    Url TEXT,
    UrlLocation TEXT,
    UrlDomain TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/EmailUrlInfo.csv'
INTO TABLE EmailUrlInfo
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE Heartbeat (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    MG TEXT,
    ManagementGroupName TEXT,
    SourceComputerId TEXT,
    ComputerIP TEXT,
    Computer TEXT,
    Category TEXT,
    OSType TEXT,
    OSName TEXT,
    OSMajorVersion TEXT,
    OSMinorVersion TEXT,
    Version TEXT,
    SCAgentChannel TEXT,
    IsGatewayInstalled TEXT,
    RemoteIPLongitude TEXT,
    RemoteIPLatitude TEXT,
    RemoteIPCountry TEXT,
    SubscriptionId TEXT,
    ResourceGroup TEXT,
    ResourceProvider TEXT,
    Resource TEXT,
    ResourceId TEXT,
    ResourceType TEXT,
    ComputerEnvironment TEXT,
    Solutions TEXT,
    VMUUID TEXT,
    ComputerPrivateIPs TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/Heartbeat.csv'
INTO TABLE Heartbeat
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE IdentityDirectoryEvents (
    TenantId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    ActionType TEXT,
    Application TEXT,
    Protocol TEXT,
    AccountName TEXT,
    AccountDomain TEXT,
    AccountUpn TEXT,
    AccountSid TEXT,
    AccountObjectId TEXT,
    AccountDisplayName TEXT,
    DeviceName TEXT,
    IPAddress TEXT,
    Port TEXT,
    DestinationDeviceName TEXT,
    DestinationIPAddress TEXT,
    DestinationPort TEXT,
    TargetDeviceName TEXT,
    TargetAccountUpn TEXT,
    TargetAccountDisplayName TEXT,
    Location TEXT,
    ISP TEXT,
    ReportId TEXT,
    AdditionalFields TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/IdentityDirectoryEvents.csv'
INTO TABLE IdentityDirectoryEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE IdentityLogonEvents (
    TenantId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    ActionType TEXT,
    Application TEXT,
    LogonType TEXT,
    Protocol TEXT,
    FailureReason TEXT,
    AccountName TEXT,
    AccountDomain TEXT,
    AccountUpn TEXT,
    AccountSid TEXT,
    AccountObjectId TEXT,
    AccountDisplayName TEXT,
    DeviceName TEXT,
    DeviceType TEXT,
    OSPlatform TEXT,
    IPAddress TEXT,
    Port TEXT,
    DestinationDeviceName TEXT,
    DestinationIPAddress TEXT,
    DestinationPort TEXT,
    TargetDeviceName TEXT,
    TargetAccountDisplayName TEXT,
    Location TEXT,
    ISP TEXT,
    ReportId TEXT,
    AdditionalFields TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/IdentityLogonEvents.csv'
INTO TABLE IdentityLogonEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE IdentityQueryEvents (
    TenantId TEXT,
    TimeGenerated TEXT,
    Timestamp TEXT,
    ActionType TEXT,
    Application TEXT,
    QueryType TEXT,
    QueryTarget TEXT,
    Query TEXT,
    Protocol TEXT,
    AccountName TEXT,
    AccountDomain TEXT,
    AccountUpn TEXT,
    AccountSid TEXT,
    AccountObjectId TEXT,
    AccountDisplayName TEXT,
    DeviceName TEXT,
    IPAddress TEXT,
    Port TEXT,
    DestinationDeviceName TEXT,
    DestinationIPAddress TEXT,
    DestinationPort TEXT,
    TargetDeviceName TEXT,
    TargetAccountUpn TEXT,
    TargetAccountDisplayName TEXT,
    Location TEXT,
    ReportId TEXT,
    AdditionalFields TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/IdentityQueryEvents.csv'
INTO TABLE IdentityQueryEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE MicrosoftAzureBastionAuditLogs (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    Time TEXT,
    OperationName TEXT,
    Location TEXT,
    UserAgent TEXT,
    UserName TEXT,
    ClientIpAddress TEXT,
    ClientPort TEXT,
    Protocol TEXT,
    ResourceType TEXT,
    TargetResourceId TEXT,
    Message TEXT,
    TargetVMIPAddress TEXT,
    UserEmail TEXT,
    TunnelId TEXT,
    SessionStartTime TEXT,
    SessionEndTime TEXT,
    Duration TEXT,
    Type TEXT,
    _ResourceId TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/MicrosoftAzureBastionAuditLogs.csv'
INTO TABLE MicrosoftAzureBastionAuditLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE MicrosoftGraphActivityLogs (
    TenantId TEXT,
    TimeGenerated TEXT,
    Location TEXT,
    RequestId TEXT,
    OperationId TEXT,
    ClientRequestId TEXT,
    ApiVersion TEXT,
    RequestMethod TEXT,
    ResponseStatusCode TEXT,
    AadTenantId TEXT,
    IPAddress TEXT,
    UserAgent TEXT,
    RequestUri TEXT,
    DurationMs TEXT,
    ResponseSizeBytes TEXT,
    SignInActivityId TEXT,
    Roles TEXT,
    TokenIssuedAt TEXT,
    AppId TEXT,
    UserId TEXT,
    ServicePrincipalId TEXT,
    Scopes TEXT,
    IdentityProvider TEXT,
    ClientAuthMethod TEXT,
    Wids TEXT,
    ATContent TEXT,
    ATContentH TEXT,
    ATContentP TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/MicrosoftGraphActivityLogs.csv'
INTO TABLE MicrosoftGraphActivityLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE SecurityAlert (
    TenantId TEXT,
    TimeGenerated TEXT,
    DisplayName TEXT,
    AlertName TEXT,
    AlertSeverity TEXT,
    Description TEXT,
    ProviderName TEXT,
    VendorName TEXT,
    VendorOriginalId TEXT,
    SystemAlertId TEXT,
    ResourceId TEXT,
    SourceComputerId TEXT,
    AlertType TEXT,
    ConfidenceLevel TEXT,
    ConfidenceScore TEXT,
    IsIncident TEXT,
    StartTime TEXT,
    EndTime TEXT,
    ProcessingEndTime TEXT,
    RemediationSteps TEXT,
    ExtendedProperties TEXT,
    Entities TEXT,
    SourceSystem TEXT,
    WorkspaceSubscriptionId TEXT,
    WorkspaceResourceGroup TEXT,
    ExtendedLinks TEXT,
    ProductName TEXT,
    ProductComponentName TEXT,
    AlertLink TEXT,
    Status TEXT,
    CompromisedEntity TEXT,
    Tactics TEXT,
    Techniques TEXT,
    SubTechniques TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/SecurityAlert.csv'
INTO TABLE SecurityAlert
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE SecurityIncident (
    TenantId TEXT,
    TimeGenerated TEXT,
    IncidentName TEXT,
    Title TEXT,
    Description TEXT,
    Severity TEXT,
    Status TEXT,
    Classification TEXT,
    ClassificationComment TEXT,
    ClassificationReason TEXT,
    Owner TEXT,
    ProviderName TEXT,
    ProviderIncidentId TEXT,
    FirstActivityTime TEXT,
    LastActivityTime TEXT,
    FirstModifiedTime TEXT,
    LastModifiedTime TEXT,
    CreatedTime TEXT,
    ClosedTime TEXT,
    IncidentNumber TEXT,
    RelatedAnalyticRuleIds TEXT,
    AlertIds TEXT,
    BookmarkIds TEXT,
    Comments TEXT,
    Tasks TEXT,
    Labels TEXT,
    IncidentUrl TEXT,
    AdditionalData TEXT,
    ModifiedBy TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/SecurityIncident.csv'
INTO TABLE SecurityIncident
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE SigninLogs (
    TenantId TEXT,
    SourceSystem TEXT,
    TimeGenerated TEXT,
    ResourceId TEXT,
    OperationName TEXT,
    OperationVersion TEXT,
    Category TEXT,
    ResultType TEXT,
    ResultSignature TEXT,
    ResultDescription TEXT,
    DurationMs TEXT,
    CorrelationId TEXT,
    Resource TEXT,
    ResourceGroup TEXT,
    ResourceProvider TEXT,
    Identity TEXT,
    Level TEXT,
    Location TEXT,
    AlternateSignInName TEXT,
    AppDisplayName TEXT,
    AppId TEXT,
    AuthenticationContextClassReferences TEXT,
    AuthenticationDetails TEXT,
    AppliedEventListeners TEXT,
    AuthenticationMethodsUsed TEXT,
    AuthenticationProcessingDetails TEXT,
    AuthenticationRequirement TEXT,
    AuthenticationRequirementPolicies TEXT,
    ClientAppUsed TEXT,
    ConditionalAccessPolicies TEXT,
    ConditionalAccessStatus TEXT,
    CreatedDateTime TEXT,
    DeviceDetail TEXT,
    IsInteractive TEXT,
    Id TEXT,
    IPAddress TEXT,
    IsRisky TEXT,
    LocationDetails TEXT,
    MfaDetail TEXT,
    NetworkLocationDetails TEXT,
    OriginalRequestId TEXT,
    ProcessingTimeInMilliseconds TEXT,
    RiskDetail TEXT,
    RiskEventTypes TEXT,
    RiskEventTypes_V2 TEXT,
    RiskLevelAggregated TEXT,
    RiskLevelDuringSignIn TEXT,
    RiskState TEXT,
    ResourceDisplayName TEXT,
    ResourceIdentity TEXT,
    ResourceServicePrincipalId TEXT,
    ServicePrincipalId TEXT,
    ServicePrincipalName TEXT,
    Status TEXT,
    TokenIssuerName TEXT,
    TokenIssuerType TEXT,
    UserAgent TEXT,
    UserDisplayName TEXT,
    UserId TEXT,
    UserPrincipalName TEXT,
    AADTenantId TEXT,
    UserType TEXT,
    FlaggedForReview TEXT,
    IPAddressFromResourceProvider TEXT,
    SignInIdentifier TEXT,
    SignInIdentifierType TEXT,
    ResourceTenantId TEXT,
    HomeTenantId TEXT,
    UniqueTokenIdentifier TEXT,
    SessionLifetimePolicies TEXT,
    AutonomousSystemNumber TEXT,
    AuthenticationProtocol TEXT,
    CrossTenantAccessType TEXT,
    AppliedConditionalAccessPolicies TEXT,
    RiskLevel TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/SigninLogs.csv'
INTO TABLE SigninLogs
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE ThreatIntelligenceIndicator (
    TenantId TEXT,
    TimeGenerated TEXT,
    SourceSystem TEXT,
    Action TEXT,
    ActivityGroupNames TEXT,
    AdditionalInformation TEXT,
    ApplicationId TEXT,
    AzureTenantId TEXT,
    ConfidenceScore TEXT,
    Description TEXT,
    DiamondModel TEXT,
    ExternalIndicatorId TEXT,
    ExpirationDateTime TEXT,
    IndicatorId TEXT,
    ThreatType TEXT,
    Active TEXT,
    KillChainActions TEXT,
    KillChainC2 TEXT,
    KillChainDelivery TEXT,
    KillChainExploitation TEXT,
    KillChainReconnaissance TEXT,
    KillChainWeaponization TEXT,
    KnownFalsePositives TEXT,
    MalwareNames TEXT,
    PassiveOnly TEXT,
    ThreatSeverity TEXT,
    Tags TEXT,
    TrafficLightProtocolLevel TEXT,
    EmailEncoding TEXT,
    EmailLanguage TEXT,
    EmailRecipient TEXT,
    EmailSenderAddress TEXT,
    EmailSenderName TEXT,
    EmailSourceDomain TEXT,
    EmailSourceIpAddress TEXT,
    EmailSubject TEXT,
    EmailXMailer TEXT,
    FileCompileDateTime TEXT,
    FileCreatedDateTime TEXT,
    FileHashType TEXT,
    FileHashValue TEXT,
    FileMutexName TEXT,
    FileName TEXT,
    FilePacker TEXT,
    FilePath TEXT,
    FileSize TEXT,
    FileType TEXT,
    DomainName TEXT,
    NetworkIP TEXT,
    NetworkPort TEXT,
    NetworkDestinationAsn TEXT,
    NetworkDestinationCidrBlock TEXT,
    NetworkDestinationIP TEXT,
    NetworkCidrBlock TEXT,
    NetworkDestinationPort TEXT,
    NetworkProtocol TEXT,
    NetworkSourceAsn TEXT,
    NetworkSourceCidrBlock TEXT,
    NetworkSourceIP TEXT,
    NetworkSourcePort TEXT,
    Url TEXT,
    UserAgent TEXT,
    IndicatorProvider TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/ThreatIntelligenceIndicator.csv'
INTO TABLE ThreatIntelligenceIndicator
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE UrlClickEvents (
    TenantId TEXT,
    Timestamp TEXT,
    TimeGenerated TEXT,
    Url TEXT,
    ActionType TEXT,
    AccountUpn TEXT,
    Workload TEXT,
    NetworkMessageId TEXT,
    IPAddress TEXT,
    ThreatTypes TEXT,
    DetectionMethods TEXT,
    IsClickedThrough TEXT,
    UrlChain TEXT,
    ReportId TEXT,
    SourceSystem TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/UrlClickEvents.csv'
INTO TABLE UrlClickEvents
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;


CREATE TABLE `Usage` (
    TenantId TEXT,
    Computer TEXT,
    TimeGenerated TEXT,
    SourceSystem TEXT,
    StartTime TEXT,
    EndTime TEXT,
    ResourceUri TEXT,
    LinkedResourceUri TEXT,
    DataType TEXT,
    Solution TEXT,
    BatchesWithinSla TEXT,
    BatchesOutsideSla TEXT,
    BatchesCapped TEXT,
    TotalBatches TEXT,
    AvgLatencyInSeconds TEXT,
    Quantity TEXT,
    QuantityUnit TEXT,
    IsBillable TEXT,
    MeterId TEXT,
    LinkedMeterId TEXT,
    Type TEXT
);


LOAD DATA INFILE '/var/lib/mysql-files/Usage.csv'
INTO TABLE `Usage`
FIELDS TERMINATED BY '❖'
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
