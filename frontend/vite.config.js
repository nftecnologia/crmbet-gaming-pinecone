import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react({
      // Enable React optimizations
      jsxRuntime: 'automatic',
      jsxImportSource: '@emotion/react',
      babel: {
        plugins: [
          // Tree shaking for lodash
          ['babel-plugin-lodash', { id: ['lodash', 'lodash-es'] }]
        ]
      }
    }),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
      manifest: {
        name: 'CRM BET Dashboard',
        short_name: 'CRM BET',
        description: 'Dashboard operacional para CRM ML com performance extrema',
        theme_color: '#1e40af',
        background_color: '#ffffff',
        display: 'standalone',
        orientation: 'portrait-primary',
        scope: '/',
        start_url: '/',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\./i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'api-cache',
              expiration: {
                maxEntries: 500,
                maxAgeSeconds: 60 * 60 * 24, // 24 hours
              },
              cacheKeyWillBeUsed: async ({ request }) => {
                return `${request.url}?timestamp=${new Date().toDateString()}`
              }
            }
          },
          {
            urlPattern: /^https:\/\/cdn\./i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'cdn-cache',
              expiration: {
                maxEntries: 200,
                maxAgeSeconds: 60 * 60 * 24 * 30, // 30 days
              }
            }
          }
        ]
      }
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: mode === 'development',
    // Ultra-optimized bundle splitting
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor chunk for large libraries
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('react-dom')) {
              return 'react-vendor'
            }
            if (id.includes('chart') || id.includes('d3')) {
              return 'chart-vendor'
            }
            if (id.includes('lodash') || id.includes('date-fns')) {
              return 'utils-vendor'
            }
            if (id.includes('framer-motion')) {
              return 'animation-vendor'
            }
            return 'vendor'
          }
          
          // Page-based code splitting
          if (id.includes('/pages/')) {
            const pageName = id.split('/pages/')[1].split('.')[0].toLowerCase()
            return `page-${pageName}`
          }
          
          // Component chunks for heavy components
          if (id.includes('VirtualTable') || id.includes('DataVisualization')) {
            return 'heavy-components'
          }
          
          // Service chunks
          if (id.includes('/services/')) {
            return 'services'
          }
        }
      }
    },
    // Optimize bundle size
    target: 'esnext',
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: mode === 'production',
        drop_debugger: mode === 'production',
        pure_funcs: mode === 'production' ? ['console.log', 'console.debug'] : []
      }
    },
    // Increase chunk size warning limit for large datasets
    chunkSizeWarningLimit: 1000
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'zustand',
      'axios',
      'chart.js',
      'react-window',
      'react-virtualized-auto-sizer'
    ],
    exclude: ['@vite/client', '@vite/env']
  },
  // Performance monitoring
  define: {
    __PERFORMANCE_MONITORING__: mode === 'production',
    __BUILD_TIME__: JSON.stringify(new Date().toISOString())
  }
}))