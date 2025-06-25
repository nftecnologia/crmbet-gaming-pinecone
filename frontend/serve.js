#!/usr/bin/env node
/**
 * 🚀 PRODUCTION FRONTEND SERVER
 * Ultra-performance static serving com cache otimizado
 */

const express = require('express');
const path = require('path');
const compression = require('compression');
const helmet = require('helmet');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
      fontSrc: ["'self'", "https://fonts.gstatic.com"],
      imgSrc: ["'self'", "data:", "https:"],
      scriptSrc: ["'self'"],
      connectSrc: ["'self'", process.env.REACT_APP_API_URL, process.env.REACT_APP_WS_URL]
    }
  }
}));

// CORS configuration
app.use(cors({
  origin: [
    'https://localhost:3000',
    'https://*.railway.app',
    process.env.REACT_APP_API_URL
  ].filter(Boolean),
  credentials: true
}));

// Compression middleware for better performance
app.use(compression({
  level: 6,
  threshold: 1024,
  filter: (req, res) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return compression.filter(req, res);
  }
}));

// Static files serving with aggressive caching
app.use(express.static(path.join(__dirname, 'build'), {
  maxAge: '1y',
  etag: true,
  lastModified: true,
  setHeaders: (res, filePath) => {
    // Cache static assets aggressively
    if (filePath.includes('/static/')) {
      res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
    }
    // HTML files should not be cached
    if (filePath.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    }
  }
}));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'frontend',
    version: '2.0.0',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// API proxy for development (optional)
if (process.env.NODE_ENV !== 'production' && process.env.REACT_APP_API_URL) {
  const { createProxyMiddleware } = require('http-proxy-middleware');
  
  app.use('/api', createProxyMiddleware({
    target: process.env.REACT_APP_API_URL,
    changeOrigin: true,
    pathRewrite: {
      '^/api': ''
    }
  }));
}

// Catch all handler for SPA routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// Error handling
app.use((err, req, res, next) => {
  console.error('Frontend server error:', err);
  res.status(500).json({ 
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'production' ? 'Something went wrong' : err.message
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  process.exit(0);
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Frontend server running on port ${PORT}`);
  console.log(`🌐 Environment: ${process.env.NODE_ENV}`);
  console.log(`📊 API URL: ${process.env.REACT_APP_API_URL}`);
  console.log(`🔗 WebSocket URL: ${process.env.REACT_APP_WS_URL}`);
});