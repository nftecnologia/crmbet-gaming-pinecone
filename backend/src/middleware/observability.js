/**
 * ENTERPRISE OBSERVABILITY MIDDLEWARE
 * 
 * Sistema completo de observabilidade para sistemas financeiros críticos
 * - Distributed Tracing com Jaeger
 * - Custom Business Metrics
 * - Real-time Performance Monitoring
 * - SLA/SLI tracking automático
 * 
 * @author DevOps Observability Team
 * @version 1.0.0
 * @monitoring CRITICAL
 */

const crypto = require('crypto');
const os = require('os');
const { performance } = require('perf_hooks');
const logger = require('../utils/logger');
const { cache } = require('../config/redis');

// Observability Configuration
const OBSERVABILITY_CONFIG = {
  // Tracing
  JAEGER_ENDPOINT: process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces',
  SERVICE_NAME: process.env.SERVICE_NAME || 'crmbet-backend',
  SERVICE_VERSION: process.env.SERVICE_VERSION || '1.0.0',
  
  // Metrics
  METRICS_INTERVAL: parseInt(process.env.METRICS_INTERVAL) || 30000, // 30s
  HEALTH_CHECK_INTERVAL: parseInt(process.env.HEALTH_CHECK_INTERVAL) || 10000, // 10s
  
  // Performance
  SLOW_QUERY_THRESHOLD: parseInt(process.env.SLOW_QUERY_THRESHOLD) || 1000, // 1s
  MEMORY_THRESHOLD: parseInt(process.env.MEMORY_THRESHOLD) || 500 * 1024 * 1024, // 500MB
  CPU_THRESHOLD: parseFloat(process.env.CPU_THRESHOLD) || 80.0, // 80%
  
  // Business Metrics
  REVENUE_TRACKING: process.env.REVENUE_TRACKING === 'true',
  USER_BEHAVIOR_TRACKING: process.env.USER_BEHAVIOR_TRACKING === 'true',
  CAMPAIGN_METRICS: process.env.CAMPAIGN_METRICS === 'true',
};

/**
 * DISTRIBUTED TRACING SYSTEM
 */
class DistributedTracer {
  constructor() {
    this.traces = new Map();
    this.spans = new Map();
    this.activeSpans = new Map();
    this.serviceName = OBSERVABILITY_CONFIG.SERVICE_NAME;
    this.serviceVersion = OBSERVABILITY_CONFIG.SERVICE_VERSION;
  }

  generateTraceId() {
    return crypto.randomBytes(16).toString('hex');
  }

  generateSpanId() {
    return crypto.randomBytes(8).toString('hex');
  }

  startTrace(operationName, parentTraceId = null) {
    const traceId = parentTraceId || this.generateTraceId();
    const spanId = this.generateSpanId();
    
    const trace = {
      traceId,
      spanId,
      parentSpanId: null,
      operationName,
      serviceName: this.serviceName,
      serviceVersion: this.serviceVersion,
      startTime: Date.now(),
      startTimeHR: process.hrtime.bigint(),
      tags: new Map(),
      logs: [],
      status: 'active'
    };

    this.traces.set(traceId, trace);
    this.activeSpans.set(traceId, spanId);
    
    return { traceId, spanId };
  }

  createSpan(traceId, operationName, parentSpanId = null) {
    const spanId = this.generateSpanId();
    const trace = this.traces.get(traceId);
    
    if (!trace) {
      logger.warn('Creating span for non-existent trace', { traceId });
      return this.startTrace(operationName);
    }

    const span = {
      traceId,
      spanId,
      parentSpanId: parentSpanId || this.activeSpans.get(traceId),
      operationName,
      serviceName: this.serviceName,
      serviceVersion: this.serviceVersion,
      startTime: Date.now(),
      startTimeHR: process.hrtime.bigint(),
      tags: new Map(),
      logs: [],
      status: 'active'
    };

    this.spans.set(spanId, span);
    this.activeSpans.set(traceId, spanId);
    
    return { traceId, spanId };
  }

  addTag(spanId, key, value) {
    const span = this.spans.get(spanId) || this.traces.get(spanId);
    if (span) {
      span.tags.set(key, value);
    }
  }

  addLog(spanId, message, data = {}) {
    const span = this.spans.get(spanId) || this.traces.get(spanId);
    if (span) {
      span.logs.push({
        timestamp: Date.now(),
        message,
        data
      });
    }
  }

  finishSpan(spanId, status = 'ok', error = null) {
    const span = this.spans.get(spanId) || this.traces.get(spanId);
    if (!span) return;

    span.endTime = Date.now();
    span.endTimeHR = process.hrtime.bigint();
    span.duration = span.endTime - span.startTime;
    span.durationHR = Number(span.endTimeHR - span.startTimeHR) / 1000000; // Convert to ms
    span.status = status;
    
    if (error) {
      span.error = {
        message: error.message,
        stack: error.stack,
        name: error.name
      };
      span.tags.set('error', true);
      span.tags.set('error.kind', error.name);
    }

    // Send to Jaeger
    this.sendToJaeger(span);
    
    // Update active span
    if (span.parentSpanId) {
      this.activeSpans.set(span.traceId, span.parentSpanId);
    }
  }

  async sendToJaeger(span) {
    try {
      const jaegerSpan = this.convertToJaegerFormat(span);
      
      // In production, send to actual Jaeger endpoint
      if (process.env.NODE_ENV === 'production') {
        // Implementation for actual Jaeger sending
        await this.postToJaeger(jaegerSpan);
      } else {
        // Log for development
        logger.trace('Jaeger Span', jaegerSpan);
      }
      
      // Store in Redis for real-time monitoring
      await cache.lpush(
        `traces:${span.traceId}`, 
        JSON.stringify(jaegerSpan),
        'EX', 3600 // Expire after 1 hour
      );
      
    } catch (error) {
      logger.error('Failed to send span to Jaeger:', error);
    }
  }

  convertToJaegerFormat(span) {
    return {
      traceID: span.traceId,
      spanID: span.spanId,
      parentSpanID: span.parentSpanId,
      operationName: span.operationName,
      startTime: span.startTime * 1000, // Jaeger expects microseconds
      duration: span.durationHR * 1000,
      tags: Array.from(span.tags.entries()).map(([key, value]) => ({
        key,
        type: typeof value === 'string' ? 'string' : 'number',
        value: String(value)
      })),
      logs: span.logs.map(log => ({
        timestamp: log.timestamp * 1000,
        fields: Object.entries(log.data).map(([key, value]) => ({
          key,
          value: String(value)
        }))
      })),
      process: {
        serviceName: span.serviceName,
        tags: [
          { key: 'version', value: span.serviceVersion },
          { key: 'hostname', value: os.hostname() },
          { key: 'pid', value: String(process.pid) }
        ]
      }
    };
  }

  async postToJaeger(span) {
    // Implementation would use HTTP client to post to Jaeger
    // For now, just log
    logger.debug('Would send to Jaeger:', { 
      endpoint: OBSERVABILITY_CONFIG.JAEGER_ENDPOINT,
      span: span.operationName 
    });
  }
}

/**
 * CUSTOM METRICS COLLECTOR
 */
class MetricsCollector {
  constructor() {
    this.metrics = new Map();
    this.businessMetrics = new Map();
    this.performanceMetrics = new Map();
    this.systemMetrics = new Map();
    
    this.startSystemMetricsCollection();
  }

  // Counter metrics
  incrementCounter(name, value = 1, tags = {}) {
    const key = this.getMetricKey(name, tags);
    const existing = this.metrics.get(key) || { type: 'counter', value: 0, tags, timestamp: Date.now() };
    existing.value += value;
    existing.timestamp = Date.now();
    this.metrics.set(key, existing);
  }

  // Gauge metrics
  setGauge(name, value, tags = {}) {
    const key = this.getMetricKey(name, tags);
    this.metrics.set(key, {
      type: 'gauge',
      value,
      tags,
      timestamp: Date.now()
    });
  }

  // Histogram metrics
  recordHistogram(name, value, tags = {}) {
    const key = this.getMetricKey(name, tags);
    const existing = this.metrics.get(key) || {
      type: 'histogram',
      values: [],
      count: 0,
      sum: 0,
      tags,
      timestamp: Date.now()
    };
    
    existing.values.push(value);
    existing.count++;
    existing.sum += value;
    existing.timestamp = Date.now();
    
    // Keep only last 100 values for memory efficiency
    if (existing.values.length > 100) {
      existing.values = existing.values.slice(-100);
    }
    
    this.metrics.set(key, existing);
  }

  // Business metrics
  recordBusinessMetric(category, name, value, metadata = {}) {
    const key = `business.${category}.${name}`;
    const metric = {
      category,
      name,
      value,
      metadata,
      timestamp: Date.now()
    };
    
    this.businessMetrics.set(key, metric);
    
    // Store in time-series format
    this.recordTimeSeries(`business_${category}_${name}`, value, metadata);
  }

  recordTimeSeries(name, value, tags = {}) {
    const timestamp = Date.now();
    const dataPoint = {
      name,
      value,
      tags,
      timestamp
    };
    
    // Store in Redis for time-series analysis
    cache.zadd(`timeseries:${name}`, timestamp, JSON.stringify(dataPoint));
    
    // Clean old data (keep last 24 hours)
    const oneDayAgo = timestamp - (24 * 60 * 60 * 1000);
    cache.zremrangebyscore(`timeseries:${name}`, 0, oneDayAgo);
  }

  getMetricKey(name, tags) {
    if (Object.keys(tags).length === 0) return name;
    
    const tagString = Object.entries(tags)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([k, v]) => `${k}=${v}`)
      .join(',');
    
    return `${name}{${tagString}}`;
  }

  // System metrics collection
  startSystemMetricsCollection() {
    setInterval(() => {
      this.collectSystemMetrics();
    }, OBSERVABILITY_CONFIG.METRICS_INTERVAL);
  }

  collectSystemMetrics() {
    const memUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    // Memory metrics
    this.setGauge('system_memory_heap_used', memUsage.heapUsed);
    this.setGauge('system_memory_heap_total', memUsage.heapTotal);
    this.setGauge('system_memory_external', memUsage.external);
    this.setGauge('system_memory_rss', memUsage.rss);
    
    // CPU metrics
    this.recordHistogram('system_cpu_user', cpuUsage.user);
    this.recordHistogram('system_cpu_system', cpuUsage.system);
    
    // System load
    const loadAvg = os.loadavg();
    this.setGauge('system_load_1m', loadAvg[0]);
    this.setGauge('system_load_5m', loadAvg[1]);
    this.setGauge('system_load_15m', loadAvg[2]);
    
    // System info
    this.setGauge('system_uptime', process.uptime());
    this.setGauge('system_free_memory', os.freemem());
    this.setGauge('system_total_memory', os.totalmem());
  }

  // Get metrics in Prometheus format
  getPrometheusMetrics() {
    const lines = [];
    
    for (const [key, metric] of this.metrics) {
      const name = key.split('{')[0];
      const tags = key.includes('{') ? key.split('{')[1].replace('}', '') : '';
      
      switch (metric.type) {
        case 'counter':
          lines.push(`# TYPE ${name} counter`);
          lines.push(`${name}${tags ? `{${tags}}` : ''} ${metric.value}`);
          break;
          
        case 'gauge':
          lines.push(`# TYPE ${name} gauge`);
          lines.push(`${name}${tags ? `{${tags}}` : ''} ${metric.value}`);
          break;
          
        case 'histogram':
          lines.push(`# TYPE ${name} histogram`);
          lines.push(`${name}_count${tags ? `{${tags}}` : ''} ${metric.count}`);
          lines.push(`${name}_sum${tags ? `{${tags}}` : ''} ${metric.sum}`);
          
          // Calculate percentiles
          const sorted = metric.values.sort((a, b) => a - b);
          const p50 = this.percentile(sorted, 0.5);
          const p95 = this.percentile(sorted, 0.95);
          const p99 = this.percentile(sorted, 0.99);
          
          lines.push(`${name}_bucket{le="0.1"${tags ? `,${tags}` : ''}} ${sorted.filter(v => v <= 0.1).length}`);
          lines.push(`${name}_bucket{le="0.5"${tags ? `,${tags}` : ''}} ${sorted.filter(v => v <= 0.5).length}`);
          lines.push(`${name}_bucket{le="1.0"${tags ? `,${tags}` : ''}} ${sorted.filter(v => v <= 1.0).length}`);
          lines.push(`${name}_bucket{le="+Inf"${tags ? `,${tags}` : ''}} ${metric.count}`);
          break;
      }
    }
    
    return lines.join('\n');
  }

  percentile(arr, p) {
    if (arr.length === 0) return 0;
    const index = Math.ceil(arr.length * p) - 1;
    return arr[Math.max(0, index)];
  }
}

/**
 * PERFORMANCE MONITOR
 */
class PerformanceMonitor {
  constructor() {
    this.slowQueries = [];
    this.performanceAlerts = [];
    this.healthStatus = 'healthy';
    
    this.startHealthChecks();
  }

  startHealthChecks() {
    setInterval(() => {
      this.checkSystemHealth();
    }, OBSERVABILITY_CONFIG.HEALTH_CHECK_INTERVAL);
  }

  async checkSystemHealth() {
    const health = {
      timestamp: Date.now(),
      memory: this.checkMemoryHealth(),
      cpu: await this.checkCPUHealth(),
      database: await this.checkDatabaseHealth(),
      redis: await this.checkRedisHealth(),
      overall: 'healthy'
    };

    // Determine overall health
    const statuses = Object.values(health).filter(v => typeof v === 'object' && v.status);
    const unhealthy = statuses.filter(s => s.status !== 'healthy');
    
    if (unhealthy.length > 0) {
      health.overall = unhealthy.some(s => s.status === 'critical') ? 'critical' : 'degraded';
    }

    this.healthStatus = health.overall;
    
    // Store health data
    await cache.set('system:health', health, 60);
    
    // Send alerts if needed
    if (health.overall !== 'healthy') {
      this.sendHealthAlert(health);
    }
  }

  checkMemoryHealth() {
    const usage = process.memoryUsage();
    const threshold = OBSERVABILITY_CONFIG.MEMORY_THRESHOLD;
    
    if (usage.heapUsed > threshold) {
      return {
        status: 'critical',
        message: 'High memory usage',
        usage: usage.heapUsed,
        threshold
      };
    }
    
    if (usage.heapUsed > threshold * 0.8) {
      return {
        status: 'warning',
        message: 'Memory usage approaching threshold',
        usage: usage.heapUsed,
        threshold
      };
    }
    
    return {
      status: 'healthy',
      usage: usage.heapUsed,
      threshold
    };
  }

  async checkCPUHealth() {
    // Simple CPU check - in production would use more sophisticated monitoring
    const loadAvg = os.loadavg()[0];
    const threshold = OBSERVABILITY_CONFIG.CPU_THRESHOLD;
    
    if (loadAvg > threshold) {
      return {
        status: 'critical',
        message: 'High CPU load',
        load: loadAvg,
        threshold
      };
    }
    
    if (loadAvg > threshold * 0.8) {
      return {
        status: 'warning',
        message: 'CPU load approaching threshold',
        load: loadAvg,
        threshold
      };
    }
    
    return {
      status: 'healthy',
      load: loadAvg,
      threshold
    };
  }

  async checkDatabaseHealth() {
    try {
      const start = performance.now();
      const { query } = require('../config/database');
      await query('SELECT 1');
      const duration = performance.now() - start;
      
      if (duration > 1000) {
        return {
          status: 'warning',
          message: 'Slow database response',
          responseTime: duration
        };
      }
      
      return {
        status: 'healthy',
        responseTime: duration
      };
    } catch (error) {
      return {
        status: 'critical',
        message: 'Database connection failed',
        error: error.message
      };
    }
  }

  async checkRedisHealth() {
    try {
      const start = performance.now();
      await cache.ping();
      const duration = performance.now() - start;
      
      if (duration > 500) {
        return {
          status: 'warning',
          message: 'Slow Redis response',
          responseTime: duration
        };
      }
      
      return {
        status: 'healthy',
        responseTime: duration
      };
    } catch (error) {
      return {
        status: 'critical',
        message: 'Redis connection failed',
        error: error.message
      };
    }
  }

  sendHealthAlert(health) {
    const alert = {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      level: health.overall === 'critical' ? 'critical' : 'warning',
      message: `System health is ${health.overall}`,
      details: health
    };

    this.performanceAlerts.push(alert);
    
    // Keep only last 100 alerts
    if (this.performanceAlerts.length > 100) {
      this.performanceAlerts = this.performanceAlerts.slice(-100);
    }

    logger.warn('Health alert generated', alert);
  }

  recordSlowQuery(query, duration, params = {}) {
    if (duration < OBSERVABILITY_CONFIG.SLOW_QUERY_THRESHOLD) return;

    const slowQuery = {
      id: crypto.randomUUID(),
      query,
      duration,
      params,
      timestamp: Date.now(),
      stackTrace: new Error().stack
    };

    this.slowQueries.push(slowQuery);
    
    // Keep only last 50 slow queries
    if (this.slowQueries.length > 50) {
      this.slowQueries = this.slowQueries.slice(-50);
    }

    logger.warn('Slow query detected', slowQuery);
  }
}

// Initialize observability services
const tracer = new DistributedTracer();
const metricsCollector = new MetricsCollector();
const performanceMonitor = new PerformanceMonitor();

/**
 * MIDDLEWARE IMPLEMENTATIONS
 */

/**
 * Request Tracing Middleware
 */
const requestTracing = (req, res, next) => {
  const startTime = performance.now();
  
  // Extract or create trace context
  const parentTraceId = req.headers['x-trace-id'];
  const parentSpanId = req.headers['x-span-id'];
  
  const { traceId, spanId } = tracer.startTrace(
    `${req.method} ${req.path}`,
    parentTraceId
  );

  // Add trace context to request
  req.traceContext = { traceId, spanId };
  
  // Add tags
  tracer.addTag(spanId, 'http.method', req.method);
  tracer.addTag(spanId, 'http.url', req.url);
  tracer.addTag(spanId, 'http.user_agent', req.get('User-Agent'));
  tracer.addTag(spanId, 'user.id', req.user?.id || 'anonymous');
  
  // Add trace headers to response
  res.set({
    'X-Trace-Id': traceId,
    'X-Span-Id': spanId
  });

  // Capture response details
  const originalSend = res.send;
  res.send = function(data) {
    const duration = performance.now() - startTime;
    
    tracer.addTag(spanId, 'http.status_code', res.statusCode);
    tracer.addTag(spanId, 'response.size', Buffer.byteLength(data || ''));
    tracer.addTag(spanId, 'duration_ms', duration);
    
    // Log performance
    if (duration > 1000) {
      tracer.addLog(spanId, 'Slow request detected', { duration });
    }
    
    // Finish span
    const status = res.statusCode >= 400 ? 'error' : 'ok';
    tracer.finishSpan(spanId, status);
    
    // Record metrics
    metricsCollector.incrementCounter('http_requests_total', 1, {
      method: req.method,
      status_code: res.statusCode,
      endpoint: req.route?.path || req.path
    });
    
    metricsCollector.recordHistogram('http_request_duration_ms', duration, {
      method: req.method,
      endpoint: req.route?.path || req.path
    });
    
    return originalSend.call(this, data);
  };

  next();
};

/**
 * Business Metrics Middleware
 */
const businessMetrics = (req, res, next) => {
  if (!OBSERVABILITY_CONFIG.USER_BEHAVIOR_TRACKING) {
    return next();
  }

  // Track user actions
  if (req.user) {
    metricsCollector.recordBusinessMetric('user', 'action', 1, {
      userId: req.user.id,
      action: `${req.method} ${req.path}`,
      userAgent: req.get('User-Agent'),
      ip: req.ip
    });
  }

  // Track API usage
  metricsCollector.recordBusinessMetric('api', 'usage', 1, {
    endpoint: req.path,
    method: req.method,
    authenticated: !!req.user
  });

  next();
};

/**
 * Performance Monitoring Middleware
 */
const performanceMonitoring = (req, res, next) => {
  const startTime = performance.now();
  const startMemory = process.memoryUsage();

  res.on('finish', () => {
    const duration = performance.now() - startTime;
    const endMemory = process.memoryUsage();
    const memoryDelta = endMemory.heapUsed - startMemory.heapUsed;

    // Record performance metrics
    metricsCollector.recordHistogram('request_memory_delta', memoryDelta);
    metricsCollector.recordHistogram('request_duration_detailed', duration, {
      endpoint: req.route?.path || req.path,
      method: req.method
    });

    // Check for performance issues
    if (duration > 5000) { // 5 seconds
      performanceMonitor.recordSlowQuery(
        `${req.method} ${req.path}`,
        duration,
        { query: req.query, body: req.body }
      );
    }
  });

  next();
};

/**
 * Error Tracking Middleware
 */
const errorTracking = (err, req, res, next) => {
  if (req.traceContext) {
    tracer.addTag(req.traceContext.spanId, 'error', true);
    tracer.addLog(req.traceContext.spanId, 'Error occurred', {
      message: err.message,
      stack: err.stack,
      name: err.name
    });
    
    tracer.finishSpan(req.traceContext.spanId, 'error', err);
  }

  // Record error metrics
  metricsCollector.incrementCounter('errors_total', 1, {
    type: err.name || 'UnknownError',
    endpoint: req.path,
    method: req.method
  });

  next(err);
};

/**
 * Health Check Endpoint
 */
const healthCheck = async (req, res) => {
  try {
    const health = await cache.get('system:health') || {
      overall: 'unknown',
      timestamp: Date.now()
    };

    const metrics = {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      loadAverage: os.loadavg(),
      activeTraces: tracer.traces.size,
      totalMetrics: metricsCollector.metrics.size,
      healthStatus: performanceMonitor.healthStatus
    };

    res.status(health.overall === 'healthy' ? 200 : 503).json({
      status: health.overall,
      timestamp: new Date().toISOString(),
      service: {
        name: OBSERVABILITY_CONFIG.SERVICE_NAME,
        version: OBSERVABILITY_CONFIG.SERVICE_VERSION
      },
      health,
      metrics
    });
  } catch (error) {
    res.status(500).json({
      status: 'error',
      message: 'Health check failed',
      error: error.message
    });
  }
};

/**
 * Metrics Endpoint (Prometheus format)
 */
const metricsEndpoint = (req, res) => {
  try {
    const metrics = metricsCollector.getPrometheusMetrics();
    res.set('Content-Type', 'text/plain');
    res.send(metrics);
  } catch (error) {
    res.status(500).json({
      error: 'Failed to generate metrics',
      message: error.message
    });
  }
};

module.exports = {
  // Middleware
  requestTracing,
  businessMetrics,
  performanceMonitoring,
  errorTracking,
  
  // Endpoints
  healthCheck,
  metricsEndpoint,
  
  // Services
  tracer,
  metricsCollector,
  performanceMonitor,
  
  // Utilities
  createCustomSpan: (operationName, traceId) => tracer.createSpan(traceId, operationName),
  recordMetric: (name, value, tags) => metricsCollector.recordTimeSeries(name, value, tags),
  recordBusinessEvent: (category, name, value, metadata) => 
    metricsCollector.recordBusinessMetric(category, name, value, metadata),
  
  // Configuration
  OBSERVABILITY_CONFIG
};