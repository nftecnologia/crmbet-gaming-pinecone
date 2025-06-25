# CRM Inteligente - API Specification

## OpenAPI/Swagger Specification

```yaml
openapi: 3.0.3
info:
  title: CRM Inteligente API
  description: |
    Comprehensive API for the CRM Inteligente system - an ML-driven customer relationship 
    management platform for gaming and betting platforms.
    
    ## Features
    - User management and authentication
    - Real-time transaction processing
    - ML-powered user clustering and segmentation  
    - Campaign management and execution
    - Analytics and reporting
    - Multi-channel notifications
    
    ## Authentication
    This API uses JWT (JSON Web Tokens) for authentication. Include the token in the 
    Authorization header: `Authorization: Bearer {token}`
    
  version: "1.0.0"
  contact:
    name: CRM Inteligente API Support
    email: support@crmbet.com
  license:
    name: MIT
    url: https://opensource.org/licenses/MIT

servers:
  - url: https://api.crmbet.com/v1
    description: Production server
  - url: https://staging-api.crmbet.com/v1
    description: Staging server
  - url: http://localhost:3000/v1
    description: Development server

security:
  - BearerAuth: []

paths:
  # ============================================================================
  # AUTHENTICATION ENDPOINTS
  # ============================================================================
  
  /auth/register:
    post:
      tags:
        - Authentication
      summary: Register new user
      description: Create a new user account with email verification
      security: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserRegistration'
            example:
              name: "João Silva"
              email: "joao@example.com"
              phone: "+5511999999999"
              password: "SecurePass123!"
              countryCode: "BRA"
              language: "pt-BR"
      responses:
        '201':
          description: User registered successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                    example: true
                  message:
                    type: string
                    example: "User registered successfully"
                  data:
                    $ref: '#/components/schemas/UserProfile'
                  token:
                    type: string
                    example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        '400':
          $ref: '#/components/responses/BadRequest'
        '409':
          $ref: '#/components/responses/Conflict'

  /auth/login:
    post:
      tags:
        - Authentication
      summary: User login
      description: Authenticate user and return JWT token
      security: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - email
                - password
              properties:
                email:
                  type: string
                  format: email
                  example: "joao@example.com"
                password:
                  type: string
                  example: "SecurePass123!"
                rememberMe:
                  type: boolean
                  default: false
      responses:
        '200':
          description: Login successful
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                    example: true
                  data:
                    $ref: '#/components/schemas/UserProfile'
                  token:
                    type: string
                    example: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                  expiresIn:
                    type: integer
                    example: 86400
        '401':
          $ref: '#/components/responses/Unauthorized'

  /auth/refresh-token:
    post:
      tags:
        - Authentication
      summary: Refresh JWT token
      description: Get a new JWT token using refresh token
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - refreshToken
              properties:
                refreshToken:
                  type: string
                  example: "refresh_token_here"
      responses:
        '200':
          description: Token refreshed successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  token:
                    type: string
                  expiresIn:
                    type: integer

  /auth/logout:
    post:
      tags:
        - Authentication
      summary: User logout
      description: Invalidate current session and JWT token
      responses:
        '200':
          description: Logout successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SuccessResponse'

  # ============================================================================
  # USER MANAGEMENT ENDPOINTS
  # ============================================================================

  /users/profile:
    get:
      tags:
        - Users
      summary: Get current user profile
      description: Retrieve the authenticated user's profile information
      responses:
        '200':
          description: User profile retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/UserProfile'
        '401':
          $ref: '#/components/responses/Unauthorized'

    put:
      tags:
        - Users
      summary: Update user profile
      description: Update the authenticated user's profile information
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserProfileUpdate'
      responses:
        '200':
          description: Profile updated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/UserProfile'
        '400':
          $ref: '#/components/responses/BadRequest'

  /users/{userId}:
    get:
      tags:
        - Users
      summary: Get user by ID
      description: Retrieve user information by user ID (Admin only)
      security:
        - BearerAuth: [admin]
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
          example: "550e8400-e29b-41d4-a716-446655440000"
      responses:
        '200':
          description: User retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/UserProfile'
        '403':
          $ref: '#/components/responses/Forbidden'
        '404':
          $ref: '#/components/responses/NotFound'

  /users/search:
    get:
      tags:
        - Users
      summary: Search users
      description: Search users with filtering and pagination (Admin only)
      security:
        - BearerAuth: [admin]
      parameters:
        - name: q
          in: query
          description: Search query (name, email)
          schema:
            type: string
          example: "joao"
        - name: status
          in: query
          description: Filter by user status
          schema:
            type: string
            enum: [active, inactive, suspended, banned]
        - name: segment
          in: query
          description: Filter by value segment
          schema:
            type: string
            enum: [high_value, medium_value, low_value, at_risk, new_user]
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: Users retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/UserProfile'
                  pagination:
                    $ref: '#/components/schemas/Pagination'

  # ============================================================================
  # TRANSACTION ENDPOINTS
  # ============================================================================

  /transactions:
    get:
      tags:
        - Transactions
      summary: Get user transactions
      description: Retrieve transactions for the authenticated user
      parameters:
        - name: type
          in: query
          description: Filter by transaction type
          schema:
            type: string
            enum: [bet, deposit, withdrawal, bonus, commission]
        - name: gameType
          in: query
          description: Filter by game type
          schema:
            type: string
        - name: startDate
          in: query
          description: Start date filter (ISO 8601)
          schema:
            type: string
            format: date-time
        - name: endDate
          in: query
          description: End date filter (ISO 8601)
          schema:
            type: string
            format: date-time
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: Transactions retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Transaction'
                  pagination:
                    $ref: '#/components/schemas/Pagination'
                  summary:
                    $ref: '#/components/schemas/TransactionSummary'

    post:
      tags:
        - Transactions
      summary: Create transaction
      description: Record a new transaction (typically called by game platform webhooks)
      security:
        - BearerAuth: [system]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TransactionCreate'
      responses:
        '201':
          description: Transaction created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/Transaction'
        '400':
          $ref: '#/components/responses/BadRequest'

  /transactions/{transactionId}:
    get:
      tags:
        - Transactions
      summary: Get transaction details
      description: Retrieve detailed information about a specific transaction
      parameters:
        - name: transactionId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Transaction retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/Transaction'
        '404':
          $ref: '#/components/responses/NotFound'

  # ============================================================================
  # CLUSTERING & ML ENDPOINTS
  # ============================================================================

  /ml/clusters:
    get:
      tags:
        - Machine Learning
      summary: Get current user clusters
      description: Retrieve current user clustering information
      security:
        - BearerAuth: [admin, analyst]
      responses:
        '200':
          description: Clusters retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/ClusterInfo'
                  metadata:
                    type: object
                    properties:
                      modelVersion:
                        type: string
                      lastUpdated:
                        type: string
                        format: date-time
                      totalClusters:
                        type: integer

  /ml/user-cluster/{userId}:
    get:
      tags:
        - Machine Learning
      summary: Get user cluster information
      description: Get clustering information for a specific user
      security:
        - BearerAuth: [admin, analyst]
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: User cluster information retrieved
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/UserCluster'

  /ml/predictions:
    post:
      tags:
        - Machine Learning
      summary: Get ML predictions
      description: Get various ML predictions for users (churn risk, lifetime value, etc.)
      security:
        - BearerAuth: [admin, analyst]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                userIds:
                  type: array
                  items:
                    type: string
                    format: uuid
                predictionTypes:
                  type: array
                  items:
                    type: string
                    enum: [churn_risk, lifetime_value, next_best_action, conversion_probability]
      responses:
        '200':
          description: Predictions generated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/MLPrediction'

  /ml/retrain:
    post:
      tags:
        - Machine Learning
      summary: Trigger model retraining
      description: Manually trigger ML model retraining (Admin only)
      security:
        - BearerAuth: [admin]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                modelType:
                  type: string
                  enum: [clustering, churn_prediction, ltv_prediction]
                  example: "clustering"
                forceRetrain:
                  type: boolean
                  default: false
      responses:
        '202':
          description: Retraining job started
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  jobId:
                    type: string
                  estimatedCompletionTime:
                    type: string
                    format: date-time

  # ============================================================================
  # CAMPAIGN MANAGEMENT ENDPOINTS
  # ============================================================================

  /campaigns:
    get:
      tags:
        - Campaigns
      summary: Get campaigns
      description: Retrieve campaigns with filtering and pagination
      security:
        - BearerAuth: [admin, marketer]
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [draft, scheduled, running, paused, completed, cancelled]
        - name: type
          in: query
          schema:
            type: string
            enum: [email, sms, push, in_app, whatsapp]
        - name: page
          in: query
          schema:
            type: integer
            minimum: 1
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            minimum: 1
            maximum: 100
            default: 20
      responses:
        '200':
          description: Campaigns retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/Campaign'
                  pagination:
                    $ref: '#/components/schemas/Pagination'

    post:
      tags:
        - Campaigns
      summary: Create campaign
      description: Create a new marketing campaign
      security:
        - BearerAuth: [admin, marketer]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CampaignCreate'
      responses:
        '201':
          description: Campaign created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/Campaign'

  /campaigns/{campaignId}:
    get:
      tags:
        - Campaigns
      summary: Get campaign details
      description: Retrieve detailed information about a specific campaign
      security:
        - BearerAuth: [admin, marketer, analyst]
      parameters:
        - name: campaignId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Campaign retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/Campaign'

    put:
      tags:
        - Campaigns
      summary: Update campaign
      description: Update campaign details (only if not running)
      security:
        - BearerAuth: [admin, marketer]
      parameters:
        - name: campaignId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CampaignUpdate'
      responses:
        '200':
          description: Campaign updated successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/Campaign'

    delete:
      tags:
        - Campaigns
      summary: Delete campaign
      description: Delete a campaign (only if not running)
      security:
        - BearerAuth: [admin]
      parameters:
        - name: campaignId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Campaign deleted successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SuccessResponse'

  /campaigns/{campaignId}/start:
    post:
      tags:
        - Campaigns
      summary: Start campaign
      description: Start execution of a scheduled campaign
      security:
        - BearerAuth: [admin, marketer]
      parameters:
        - name: campaignId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Campaign started successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  message:
                    type: string
                  executionId:
                    type: string

  /campaigns/{campaignId}/pause:
    post:
      tags:
        - Campaigns
      summary: Pause campaign
      description: Pause a running campaign
      security:
        - BearerAuth: [admin, marketer]
      parameters:
        - name: campaignId
          in: path
          required: true
          schema:
            type: string
            format: uuid
      responses:
        '200':
          description: Campaign paused successfully

  /campaigns/{campaignId}/results:
    get:
      tags:
        - Campaigns
      summary: Get campaign results
      description: Retrieve detailed campaign performance results
      security:
        - BearerAuth: [admin, marketer, analyst]
      parameters:
        - name: campaignId
          in: path
          required: true
          schema:
            type: string
            format: uuid
        - name: includeUserDetails
          in: query
          description: Include individual user interaction details
          schema:
            type: boolean
            default: false
      responses:
        '200':
          description: Campaign results retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/CampaignResults'

  # ============================================================================
  # ANALYTICS ENDPOINTS
  # ============================================================================

  /analytics/dashboard:
    get:
      tags:
        - Analytics
      summary: Get dashboard analytics
      description: Retrieve key metrics for the analytics dashboard
      security:
        - BearerAuth: [admin, analyst]
      parameters:
        - name: timeRange
          in: query
          description: Time range for analytics
          schema:
            type: string
            enum: [24h, 7d, 30d, 90d, 1y]
            default: "7d"
      responses:
        '200':
          description: Dashboard analytics retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/DashboardAnalytics'

  /analytics/user-segments:
    get:
      tags:
        - Analytics
      summary: Get user segment analytics
      description: Retrieve analytics about user segments and clusters
      security:
        - BearerAuth: [admin, analyst]
      responses:
        '200':
          description: User segment analytics retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/SegmentAnalytics'

  /analytics/campaign-performance:
    get:
      tags:
        - Analytics
      summary: Get campaign performance analytics
      description: Retrieve campaign performance metrics and comparisons
      security:
        - BearerAuth: [admin, analyst, marketer]
      parameters:
        - name: timeRange
          in: query
          schema:
            type: string
            enum: [7d, 30d, 90d]
            default: "30d"
        - name: campaignType
          in: query
          schema:
            type: string
            enum: [email, sms, push, in_app, whatsapp]
      responses:
        '200':
          description: Campaign performance analytics retrieved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  data:
                    $ref: '#/components/schemas/CampaignAnalytics'

  # ============================================================================
  # SYSTEM ENDPOINTS
  # ============================================================================

  /health:
    get:
      tags:
        - System
      summary: Health check
      description: Check system health and service availability
      security: []
      responses:
        '200':
          description: System is healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "healthy"
                  timestamp:
                    type: string
                    format: date-time
                  services:
                    type: object
                    properties:
                      database:
                        type: string
                        enum: [healthy, unhealthy]
                      cache:
                        type: string
                        enum: [healthy, unhealthy]
                      messageQueue:
                        type: string
                        enum: [healthy, unhealthy]
                      mlService:
                        type: string
                        enum: [healthy, unhealthy]

  /metrics:
    get:
      tags:
        - System
      summary: System metrics
      description: Retrieve system performance metrics (Admin only)
      security:
        - BearerAuth: [admin]
      responses:
        '200':
          description: System metrics retrieved successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SystemMetrics'

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    # ========================================================================
    # USER SCHEMAS
    # ========================================================================
    
    UserRegistration:
      type: object
      required:
        - name
        - email
        - password
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 255
          example: "João Silva"
        email:
          type: string
          format: email
          example: "joao@example.com"
        phone:
          type: string
          pattern: '^\+[1-9]\d{10,14}$'
          example: "+5511999999999"
        password:
          type: string
          minLength: 8
          example: "SecurePass123!"
        countryCode:
          type: string
          length: 3
          example: "BRA"
        language:
          type: string
          enum: [pt-BR, en-US, es-ES]
          default: "pt-BR"
        ageRange:
          type: string
          enum: [18-25, 26-35, 36-45, 46-55, 55+]
        gender:
          type: string
          enum: [male, female, other, prefer_not_to_say]

    UserProfile:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        email:
          type: string
          format: email
        phone:
          type: string
        countryCode:
          type: string
        language:
          type: string
        timezone:
          type: string
        ageRange:
          type: string
        gender:
          type: string
        locationCity:
          type: string
        locationState:
          type: string
        preferredGames:
          type: array
          items:
            type: string
        riskTolerance:
          type: string
          enum: [low, medium, high]
        lifetimeValue:
          type: number
          format: decimal
        status:
          type: string
          enum: [active, inactive, suspended, banned]
        emailVerified:
          type: boolean
        phoneVerified:
          type: boolean
        kycStatus:
          type: string
          enum: [pending, approved, rejected]
        createdAt:
          type: string
          format: date-time
        updatedAt:
          type: string
          format: date-time
        lastLoginAt:
          type: string
          format: date-time
        lastActivityAt:
          type: string
          format: date-time
        cluster:
          $ref: '#/components/schemas/UserCluster'

    UserProfileUpdate:
      type: object
      properties:
        name:
          type: string
          minLength: 2
          maxLength: 255
        phone:
          type: string
        language:
          type: string
          enum: [pt-BR, en-US, es-ES]
        timezone:
          type: string
        ageRange:
          type: string
          enum: [18-25, 26-35, 36-45, 46-55, 55+]
        gender:
          type: string
          enum: [male, female, other, prefer_not_to_say]
        locationCity:
          type: string
        locationState:
          type: string
        preferredGames:
          type: array
          items:
            type: string
        riskTolerance:
          type: string
          enum: [low, medium, high]

    # ========================================================================
    # TRANSACTION SCHEMAS
    # ========================================================================

    Transaction:
      type: object
      properties:
        id:
          type: string
          format: uuid
        userId:
          type: string
          format: uuid
        transactionType:
          type: string
          enum: [bet, deposit, withdrawal, bonus, commission]
        gameType:
          type: string
        gameProvider:
          type: string
        gameName:
          type: string
        amount:
          type: number
          format: decimal
        currency:
          type: string
          length: 3
          default: "BRL"
        balanceBefore:
          type: number
          format: decimal
        balanceAfter:
          type: number
          format: decimal
        channel:
          type: string
          enum: [web, mobile_app, mobile_web, api]
        deviceType:
          type: string
          enum: [desktop, mobile, tablet]
        paymentMethod:
          type: string
        sessionId:
          type: string
        sessionDurationMinutes:
          type: integer
        ipAddress:
          type: string
          format: ipv4
        userAgent:
          type: string
        riskScore:
          type: number
          format: decimal
          minimum: 0
          maximum: 100
        isSuspicious:
          type: boolean
        complianceFlags:
          type: array
          items:
            type: string
        status:
          type: string
          enum: [pending, completed, failed, cancelled]
        processedAt:
          type: string
          format: date-time
        timestamp:
          type: string
          format: date-time
        externalTransactionId:
          type: string
        providerTransactionId:
          type: string

    TransactionCreate:
      type: object
      required:
        - userId
        - transactionType
        - amount
      properties:
        userId:
          type: string
          format: uuid
        transactionType:
          type: string
          enum: [bet, deposit, withdrawal, bonus, commission]
        gameType:
          type: string
        gameProvider:
          type: string
        gameName:
          type: string
        amount:
          type: number
          format: decimal
        currency:
          type: string
          length: 3
          default: "BRL"
        balanceBefore:
          type: number
          format: decimal
        balanceAfter:
          type: number
          format: decimal
        channel:
          type: string
          enum: [web, mobile_app, mobile_web, api]
        deviceType:
          type: string
          enum: [desktop, mobile, tablet]
        paymentMethod:
          type: string
        sessionId:
          type: string
        sessionDurationMinutes:
          type: integer
        ipAddress:
          type: string
          format: ipv4
        userAgent:
          type: string
        externalTransactionId:
          type: string
        providerTransactionId:
          type: string

    TransactionSummary:
      type: object
      properties:
        totalTransactions:
          type: integer
        totalAmount:
          type: number
          format: decimal
        totalBets:
          type: number
          format: decimal
        totalDeposits:
          type: number
          format: decimal
        totalWithdrawals:
          type: number
          format: decimal
        averageTransactionAmount:
          type: number
          format: decimal
        uniqueGameTypes:
          type: integer
        lastTransactionAt:
          type: string
          format: date-time

    # ========================================================================
    # ML & CLUSTERING SCHEMAS
    # ========================================================================

    UserCluster:
      type: object
      properties:
        id:
          type: string
          format: uuid
        userId:
          type: string
          format: uuid
        clusterId:
          type: integer
        clusterName:
          type: string
        clusterDescription:
          type: string
        modelVersion:
          type: string
        algorithmUsed:
          type: string
        features:
          type: object
          description: "Feature vector used for clustering"
        confidence:
          type: number
          format: decimal
          minimum: 0
          maximum: 1
        distanceToCentroid:
          type: number
          format: decimal
        clusterSize:
          type: integer
        clusterCharacteristics:
          type: object
          description: "Key traits of this cluster"
        valueSegment:
          type: string
          enum: [high_value, medium_value, low_value, at_risk, new_user]
        behaviorPattern:
          type: string
        churnRisk:
          type: number
          format: decimal
          minimum: 0
          maximum: 1
        validFrom:
          type: string
          format: date-time
        validTo:
          type: string
          format: date-time
        isCurrent:
          type: boolean
        createdAt:
          type: string
          format: date-time
        updatedAt:
          type: string
          format: date-time

    ClusterInfo:
      type: object
      properties:
        clusterId:
          type: integer
        clusterName:
          type: string
        description:
          type: string
        size:
          type: integer
        characteristics:
          type: object
        averageLifetimeValue:
          type: number
          format: decimal
        averageChurnRisk:
          type: number
          format: decimal
        topGameTypes:
          type: array
          items:
            type: string
        demographics:
          type: object
          properties:
            avgAge:
              type: string
            genderDistribution:
              type: object
            topCountries:
              type: array
              items:
                type: string

    MLPrediction:
      type: object
      properties:
        userId:
          type: string
          format: uuid
        predictionType:
          type: string
          enum: [churn_risk, lifetime_value, next_best_action, conversion_probability]
        prediction:
          type: object
          description: "Prediction result (varies by type)"
        confidence:
          type: number
          format: decimal
          minimum: 0
          maximum: 1
        modelVersion:
          type: string
        createdAt:
          type: string
          format: date-time
        expiresAt:
          type: string
          format: date-time

    # ========================================================================
    # CAMPAIGN SCHEMAS
    # ========================================================================

    Campaign:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        description:
          type: string
        campaignType:
          type: string
          enum: [email, sms, push, in_app, whatsapp]
        clusterTarget:
          type: array
          items:
            type: integer
        userSegments:
          type: array
          items:
            type: string
        geoTarget:
          type: array
          items:
            type: string
        subjectLine:
          type: string
        message:
          type: string
        ctaText:
          type: string
        ctaUrl:
          type: string
        templateId:
          type: string
        imageUrls:
          type: array
          items:
            type: string
        videoUrl:
          type: string
        personalizationFields:
          type: object
        abTestVariant:
          type: string
        scheduledAt:
          type: string
          format: date-time
        sendTimezone:
          type: string
        budgetTotal:
          type: number
          format: decimal
        budgetSpent:
          type: number
          format: decimal
        maxSends:
          type: integer
        status:
          type: string
          enum: [draft, scheduled, running, paused, completed, cancelled]
        priority:
          type: integer
          minimum: 1
          maximum: 10
        requiresApproval:
          type: boolean
        approvedBy:
          type: string
          format: uuid
        approvedAt:
          type: string
          format: date-time
        targetAudienceSize:
          type: integer
        estimatedReach:
          type: integer
        createdAt:
          type: string
          format: date-time
        updatedAt:
          type: string
          format: date-time
        createdBy:
          type: string
          format: uuid
        updatedBy:
          type: string
          format: uuid

    CampaignCreate:
      type: object
      required:
        - name
        - campaignType
        - message
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 255
        description:
          type: string
        campaignType:
          type: string
          enum: [email, sms, push, in_app, whatsapp]
        clusterTarget:
          type: array
          items:
            type: integer
        userSegments:
          type: array
          items:
            type: string
            enum: [high_value, medium_value, low_value, at_risk, new_user]
        geoTarget:
          type: array
          items:
            type: string
        subjectLine:
          type: string
          maxLength: 200
        message:
          type: string
          minLength: 1
        ctaText:
          type: string
          maxLength: 100
        ctaUrl:
          type: string
          format: uri
        templateId:
          type: string
        imageUrls:
          type: array
          items:
            type: string
            format: uri
        videoUrl:
          type: string
          format: uri
        personalizationFields:
          type: object
        abTestVariant:
          type: string
        scheduledAt:
          type: string
          format: date-time
        sendTimezone:
          type: string
          default: "America/Sao_Paulo"
        budgetTotal:
          type: number
          format: decimal
          minimum: 0
        maxSends:
          type: integer
          minimum: 1
        priority:
          type: integer
          minimum: 1
          maximum: 10
          default: 5

    CampaignUpdate:
      type: object
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 255
        description:
          type: string
        subjectLine:
          type: string
          maxLength: 200
        message:
          type: string
          minLength: 1
        ctaText:
          type: string
          maxLength: 100
        ctaUrl:
          type: string
          format: uri
        scheduledAt:
          type: string
          format: date-time
        budgetTotal:
          type: number
          format: decimal
          minimum: 0
        maxSends:
          type: integer
          minimum: 1
        priority:
          type: integer
          minimum: 1
          maximum: 10

    CampaignResults:
      type: object
      properties:
        campaignId:
          type: string
          format: uuid
        campaignName:
          type: string
        performance:
          type: object
          properties:
            totalSent:
              type: integer
            totalDelivered:
              type: integer
            totalOpened:
              type: integer
            totalClicked:
              type: integer
            totalConverted:
              type: integer
            totalUnsubscribed:
              type: integer
            deliveryRate:
              type: number
              format: decimal
            openRate:
              type: number
              format: decimal
            clickRate:
              type: number
              format: decimal
            conversionRate:
              type: number
              format: decimal
            unsubscribeRate:
              type: number
              format: decimal
            totalRevenue:
              type: number
              format: decimal
            attributedRevenue:
              type: number
              format: decimal
            roas:
              type: number
              format: decimal
              description: "Return on Ad Spend"
        timeline:
          type: array
          items:
            type: object
            properties:
              timestamp:
                type: string
                format: date-time
              event:
                type: string
              count:
                type: integer
        segmentPerformance:
          type: array
          items:
            type: object
            properties:
              segment:
                type: string
              sent:
                type: integer
              openRate:
                type: number
                format: decimal
              clickRate:
                type: number
                format: decimal
              conversionRate:
                type: number
                format: decimal
        deviceBreakdown:
          type: object
          properties:
            desktop:
              type: number
              format: decimal
            mobile:
              type: number
              format: decimal
            tablet:
              type: number
              format: decimal

    # ========================================================================
    # ANALYTICS SCHEMAS
    # ========================================================================

    DashboardAnalytics:
      type: object
      properties:
        userMetrics:
          type: object
          properties:
            totalUsers:
              type: integer
            activeUsers:
              type: integer
            newUsers:
              type: integer
            churnedUsers:
              type: integer
            userGrowthRate:
              type: number
              format: decimal
        transactionMetrics:
          type: object
          properties:
            totalTransactions:
              type: integer
            totalVolume:
              type: number
              format: decimal
            averageTransactionValue:
              type: number
              format: decimal
            transactionGrowthRate:
              type: number
              format: decimal
        campaignMetrics:
          type: object
          properties:
            activeCampaigns:
              type: integer
            completedCampaigns:
              type: integer
            averageOpenRate:
              type: number
              format: decimal
            averageConversionRate:
              type: number
              format: decimal
            totalCampaignRevenue:
              type: number
              format: decimal
        mlMetrics:
          type: object
          properties:
            totalClusters:
              type: integer
            highRiskUsers:
              type: integer
            modelAccuracy:
              type: number
              format: decimal
            lastRetrainDate:
              type: string
              format: date-time

    SegmentAnalytics:
      type: object
      properties:
        segments:
          type: array
          items:
            type: object
            properties:
              segment:
                type: string
              userCount:
                type: integer
              averageLifetimeValue:
                type: number
                format: decimal
              averageChurnRisk:
                type: number
                format: decimal
              conversionRate:
                type: number
                format: decimal
              topGameTypes:
                type: array
                items:
                  type: string
        clusterDistribution:
          type: array
          items:
            type: object
            properties:
              clusterId:
                type: integer
              clusterName:
                type: string
              size:
                type: integer
              percentage:
                type: number
                format: decimal
        segmentTrends:
          type: array
          items:
            type: object
            properties:
              date:
                type: string
                format: date
              segment:
                type: string
              userCount:
                type: integer

    CampaignAnalytics:
      type: object
      properties:
        overview:
          type: object
          properties:
            totalCampaigns:
              type: integer
            averageOpenRate:
              type: number
              format: decimal
            averageClickRate:
              type: number
              format: decimal
            averageConversionRate:
              type: number
              format: decimal
            totalRevenue:
              type: number
              format: decimal
        byType:
          type: array
          items:
            type: object
            properties:
              campaignType:
                type: string
              count:
                type: integer
              averageOpenRate:
                type: number
                format: decimal
              averageConversionRate:
                type: number
                format: decimal
              totalRevenue:
                type: number
                format: decimal
        topPerforming:
          type: array
          items:
            type: object
            properties:
              campaignId:
                type: string
                format: uuid
              campaignName:
                type: string
              openRate:
                type: number
                format: decimal
              conversionRate:
                type: number
                format: decimal
              revenue:
                type: number
                format: decimal
        trends:
          type: array
          items:
            type: object
            properties:
              date:
                type: string
                format: date
              campaigns:
                type: integer
              opens:
                type: integer
              clicks:
                type: integer
              conversions:
                type: integer

    SystemMetrics:
      type: object
      properties:
        api:
          type: object
          properties:
            requestsPerSecond:
              type: number
            averageResponseTime:
              type: number
            errorRate:
              type: number
            activeConnections:
              type: integer
        database:
          type: object
          properties:
            connections:
              type: integer
            queriesPerSecond:
              type: number
            averageQueryTime:
              type: number
            cacheHitRatio:
              type: number
        ml:
          type: object
          properties:
            predictionsPerSecond:
              type: number
            averagePredictionTime:
              type: number
            modelAccuracy:
              type: number
            queueLength:
              type: integer
        memory:
          type: object
          properties:
            used:
              type: number
            total:
              type: number
            percentage:
              type: number
        cpu:
          type: object
          properties:
            usage:
              type: number
            cores:
              type: integer

    # ========================================================================
    # COMMON SCHEMAS
    # ========================================================================

    Pagination:
      type: object
      properties:
        page:
          type: integer
          minimum: 1
        limit:
          type: integer
          minimum: 1
          maximum: 100
        total:
          type: integer
        totalPages:
          type: integer
        hasNext:
          type: boolean
        hasPrev:
          type: boolean

    SuccessResponse:
      type: object
      properties:
        success:
          type: boolean
          example: true
        message:
          type: string
          example: "Operation completed successfully"

    ErrorResponse:
      type: object
      properties:
        success:
          type: boolean
          example: false
        error:
          type: object
          properties:
            code:
              type: string
              example: "VALIDATION_ERROR"
            message:
              type: string
              example: "Invalid input data"
            details:
              type: array
              items:
                type: object
                properties:
                  field:
                    type: string
                  message:
                    type: string

  responses:
    BadRequest:
      description: Bad request - invalid input data
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "VALIDATION_ERROR"
              message: "Invalid input data"
              details:
                - field: "email"
                  message: "Invalid email format"

    Unauthorized:
      description: Unauthorized - invalid or missing authentication
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "UNAUTHORIZED"
              message: "Invalid or missing authentication token"

    Forbidden:
      description: Forbidden - insufficient permissions
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "FORBIDDEN"
              message: "Insufficient permissions to access this resource"

    NotFound:
      description: Resource not found
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "NOT_FOUND"
              message: "Requested resource not found"

    Conflict:
      description: Conflict - resource already exists
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "CONFLICT"
              message: "Resource already exists"

    InternalServerError:
      description: Internal server error
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "INTERNAL_ERROR"
              message: "An unexpected error occurred"

    RateLimitExceeded:
      description: Rate limit exceeded
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/ErrorResponse'
          example:
            success: false
            error:
              code: "RATE_LIMIT_EXCEEDED"
              message: "Too many requests - rate limit exceeded"

tags:
  - name: Authentication
    description: User authentication and authorization endpoints
  - name: Users
    description: User management and profile operations
  - name: Transactions
    description: Transaction processing and history
  - name: Machine Learning
    description: ML clustering, predictions, and model management
  - name: Campaigns
    description: Marketing campaign management and execution
  - name: Analytics
    description: Analytics and reporting endpoints
  - name: System
    description: System health and monitoring endpoints
```

## API Usage Examples

### Authentication Flow
```javascript
// Register new user
const registerResponse = await fetch('/v1/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'João Silva',
    email: 'joao@example.com',
    password: 'SecurePass123!',
    countryCode: 'BRA'
  })
});

// Login
const loginResponse = await fetch('/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'joao@example.com',
    password: 'SecurePass123!'
  })
});

const { token } = await loginResponse.json();
```

### Creating and Managing Campaigns
```javascript
// Create a new campaign
const campaignResponse = await fetch('/v1/campaigns', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    name: 'Welcome New Users',
    campaignType: 'email',
    userSegments: ['new_user'],
    subjectLine: 'Bem-vindo ao CRMBet! 🎉',
    message: 'Olá {{name}}, seja bem-vindo(a)!',
    ctaText: 'Começar Agora',
    ctaUrl: 'https://crmbet.com/welcome',
    scheduledAt: '2024-01-15T10:00:00Z'
  })
});

// Get campaign results
const resultsResponse = await fetch(`/v1/campaigns/${campaignId}/results`, {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

### ML Predictions and Analytics
```javascript
// Get user cluster information
const clusterResponse = await fetch(`/v1/ml/user-cluster/${userId}`, {
  headers: { 'Authorization': `Bearer ${token}` }
});

// Request ML predictions
const predictionsResponse = await fetch('/v1/ml/predictions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    userIds: [userId],
    predictionTypes: ['churn_risk', 'lifetime_value']
  })
});
```

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability:

- **Authentication endpoints**: 5 requests per 15 minutes
- **General API endpoints**: 1000 requests per 15 minutes  
- **ML prediction endpoints**: 100 requests per minute
- **Webhook endpoints**: 10,000 requests per minute

Rate limit headers are included in all responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

## Error Handling

All API responses follow a consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": [
      {
        "field": "fieldName",
        "message": "Field-specific error message"
      }
    ]
  }
}
```

Common error codes:
- `VALIDATION_ERROR`: Input validation failed
- `UNAUTHORIZED`: Authentication required or invalid
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `CONFLICT`: Resource already exists
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `INTERNAL_ERROR`: Server error

## Webhook Integration

The API supports webhooks for real-time event notifications:

### Supported Events
- `user.created` - New user registration
- `user.updated` - User profile changes
- `transaction.completed` - Transaction processed
- `campaign.started` - Campaign execution started
- `campaign.completed` - Campaign execution finished
- `cluster.updated` - User clustering updated

### Webhook Payload Format
```json
{
  "event": "user.created",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "userId": "550e8400-e29b-41d4-a716-446655440000",
    "email": "user@example.com"
  },
  "signature": "sha256=..."
}
```

This comprehensive API specification provides a complete reference for integrating with the CRM Inteligente system, covering all major functionality from user management to advanced ML predictions and campaign execution.