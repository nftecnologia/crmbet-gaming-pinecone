import React, { memo, useRef, useEffect, useMemo, useCallback, useState } from 'react'
import throttle from 'lodash.throttle'
import debounce from 'lodash.debounce'
import { Download, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react'

/**
 * Ultra-Performance Canvas-Based Data Visualization
 * Optimized for datasets with 1M+ data points
 */
const DataVisualization = memo(({
  data = [],
  type = 'scatter',
  width = 800,
  height = 600,
  sampling = 'intelligent',
  maxPoints = 10000,
  interactive = true,
  showTooltip = true,
  theme = 'light',
  className = ''
}) => {
  const canvasRef = useRef(null)
  const contextRef = useRef(null)
  const animationRef = useRef(null)
  const workerRef = useRef(null)
  
  const [viewport, setViewport] = useState({
    x: 0,
    y: 0,
    scale: 1,
    minScale: 0.1,
    maxScale: 10
  })
  
  const [tooltip, setTooltip] = useState({
    visible: false,
    x: 0,
    y: 0,
    data: null
  })
  
  const [isRendering, setIsRendering] = useState(false)
  const [renderStats, setRenderStats] = useState({
    pointsRendered: 0,
    renderTime: 0,
    fps: 0
  })

  // Color schemes for different themes
  const colorSchemes = {
    light: {
      background: '#ffffff',
      grid: '#f0f0f0',
      axis: '#333333',
      points: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'],
      text: '#1f2937'
    },
    dark: {
      background: '#1f2937',
      grid: '#374151',
      axis: '#e5e7eb',
      points: ['#60a5fa', '#f87171', '#34d399', '#fbbf24', '#a78bfa'],
      text: '#f9fafb'
    }
  }

  // Initialize Web Worker for data processing
  useEffect(() => {
    const workerCode = `
      // Data sampling algorithms
      const sampleData = (data, maxPoints, method) => {
        if (data.length <= maxPoints) return data
        
        switch (method) {
          case 'uniform':
            return uniformSampling(data, maxPoints)
          case 'intelligent':
            return intelligentSampling(data, maxPoints)
          case 'clustering':
            return clusterSampling(data, maxPoints)
          default:
            return uniformSampling(data, maxPoints)
        }
      }
      
      const uniformSampling = (data, maxPoints) => {
        const step = Math.ceil(data.length / maxPoints)
        return data.filter((_, index) => index % step === 0)
      }
      
      const intelligentSampling = (data, maxPoints) => {
        if (data.length <= maxPoints) return data
        
        // Sort by importance (e.g., magnitude, variance)
        const withImportance = data.map((point, index) => ({
          ...point,
          importance: calculateImportance(point, index, data)
        }))
        
        withImportance.sort((a, b) => b.importance - a.importance)
        return withImportance.slice(0, maxPoints)
      }
      
      const calculateImportance = (point, index, data) => {
        // Calculate importance based on multiple factors
        let importance = 0
        
        // Distance from center
        const centerX = data.reduce((sum, p) => sum + p.x, 0) / data.length
        const centerY = data.reduce((sum, p) => sum + p.y, 0) / data.length
        const distance = Math.sqrt(Math.pow(point.x - centerX, 2) + Math.pow(point.y - centerY, 2))
        importance += distance * 0.3
        
        // Local variance (points that stand out)
        const neighbors = data.slice(Math.max(0, index - 5), index + 6)
        const avgX = neighbors.reduce((sum, p) => sum + p.x, 0) / neighbors.length
        const avgY = neighbors.reduce((sum, p) => sum + p.y, 0) / neighbors.length
        const variance = Math.sqrt(Math.pow(point.x - avgX, 2) + Math.pow(point.y - avgY, 2))
        importance += variance * 0.7
        
        return importance
      }
      
      const clusterSampling = (data, maxPoints) => {
        // Simple k-means clustering for representative sampling
        const clusters = Math.min(maxPoints, Math.ceil(data.length / 100))
        const centroids = []
        
        // Initialize centroids randomly
        for (let i = 0; i < clusters; i++) {
          const randomIndex = Math.floor(Math.random() * data.length)
          centroids.push({ ...data[randomIndex] })
        }
        
        // Assign points to clusters and return representatives
        const representatives = []
        centroids.forEach(centroid => {
          let closest = data[0]
          let minDistance = Infinity
          
          data.forEach(point => {
            const distance = Math.sqrt(
              Math.pow(point.x - centroid.x, 2) + 
              Math.pow(point.y - centroid.y, 2)
            )
            
            if (distance < minDistance) {
              minDistance = distance
              closest = point
            }
          })
          
          representatives.push(closest)
        })
        
        return representatives
      }
      
      // Message handler
      self.onmessage = function(e) {
        const { type, data, maxPoints, method } = e.data
        
        switch (type) {
          case 'sample':
            const sampled = sampleData(data, maxPoints, method)
            self.postMessage({ type: 'sampled', data: sampled })
            break
            
          case 'analyze':
            const stats = {
              count: data.length,
              xRange: [Math.min(...data.map(p => p.x)), Math.max(...data.map(p => p.x))],
              yRange: [Math.min(...data.map(p => p.y)), Math.max(...data.map(p => p.y))],
              density: data.length / (800 * 600) // points per pixel
            }
            self.postMessage({ type: 'analyzed', stats })
            break
        }
      }
    `
    
    const blob = new Blob([workerCode], { type: 'application/javascript' })
    workerRef.current = new Worker(URL.createObjectURL(blob))
    
    workerRef.current.onmessage = (e) => {
      const { type, data: workerData, stats } = e.data
      
      if (type === 'sampled') {
        renderVisualization(workerData)
      } else if (type === 'analyzed') {
        console.log('Data analysis:', stats)
      }
    }
    
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate()
      }
    }
  }, [])

  // Setup canvas and context
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const context = canvas.getContext('2d', {
      alpha: false,
      desynchronized: true,
      powerPreference: 'high-performance'
    })
    
    contextRef.current = context
    
    // Set canvas size with device pixel ratio for crisp rendering
    const dpr = window.devicePixelRatio || 1
    canvas.width = width * dpr
    canvas.height = height * dpr
    canvas.style.width = `${width}px`
    canvas.style.height = `${height}px`
    context.scale(dpr, dpr)
    
    // Enable image smoothing for better quality
    context.imageSmoothingEnabled = true
    context.imageSmoothingQuality = 'high'
    
  }, [width, height])

  // Process data when it changes
  useEffect(() => {
    if (data.length === 0 || !workerRef.current) return
    
    setIsRendering(true)
    
    // Send data to worker for sampling
    workerRef.current.postMessage({
      type: 'sample',
      data,
      maxPoints,
      method: sampling
    })
    
    // Also analyze data
    workerRef.current.postMessage({
      type: 'analyze',
      data
    })
    
  }, [data, sampling, maxPoints])

  // Render visualization on canvas
  const renderVisualization = useCallback((sampledData) => {
    const canvas = canvasRef.current
    const context = contextRef.current
    if (!canvas || !context || sampledData.length === 0) return

    const startTime = performance.now()
    const colors = colorSchemes[theme]
    
    // Clear canvas
    context.fillStyle = colors.background
    context.fillRect(0, 0, width, height)
    
    // Calculate data bounds
    const xMin = Math.min(...sampledData.map(p => p.x))
    const xMax = Math.max(...sampledData.map(p => p.x))
    const yMin = Math.min(...sampledData.map(p => p.y))
    const yMax = Math.max(...sampledData.map(p => p.y))
    
    // Add padding
    const padding = 40
    const plotWidth = width - 2 * padding
    const plotHeight = height - 2 * padding
    
    // Scale functions
    const scaleX = (x) => padding + ((x - xMin) / (xMax - xMin)) * plotWidth * viewport.scale + viewport.x
    const scaleY = (y) => height - padding - ((y - yMin) / (yMax - yMin)) * plotHeight * viewport.scale + viewport.y
    
    // Draw grid
    context.strokeStyle = colors.grid
    context.lineWidth = 0.5
    
    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding + (i / 10) * plotWidth
      context.beginPath()
      context.moveTo(x, padding)
      context.lineTo(x, height - padding)
      context.stroke()
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = padding + (i / 10) * plotHeight
      context.beginPath()
      context.moveTo(padding, y)
      context.lineTo(width - padding, y)
      context.stroke()
    }
    
    // Draw axes
    context.strokeStyle = colors.axis
    context.lineWidth = 2
    
    // X axis
    context.beginPath()
    context.moveTo(padding, height - padding)
    context.lineTo(width - padding, height - padding)
    context.stroke()
    
    // Y axis
    context.beginPath()
    context.moveTo(padding, padding)
    context.lineTo(padding, height - padding)
    context.stroke()
    
    // Render data points based on type
    switch (type) {
      case 'scatter':
        renderScatterPlot(context, sampledData, scaleX, scaleY, colors)
        break
      case 'line':
        renderLinePlot(context, sampledData, scaleX, scaleY, colors)
        break
      case 'heatmap':
        renderHeatmap(context, sampledData, scaleX, scaleY, colors)
        break
      case 'histogram':
        renderHistogram(context, sampledData, scaleX, scaleY, colors)
        break
      default:
        renderScatterPlot(context, sampledData, scaleX, scaleY, colors)
    }
    
    // Render axes labels
    renderLabels(context, xMin, xMax, yMin, yMax, colors)
    
    const endTime = performance.now()
    const renderTime = endTime - startTime
    
    setRenderStats({
      pointsRendered: sampledData.length,
      renderTime: Math.round(renderTime),
      fps: Math.round(1000 / renderTime)
    })
    
    setIsRendering(false)
  }, [width, height, viewport, theme, type])

  // Render scatter plot
  const renderScatterPlot = (context, data, scaleX, scaleY, colors) => {
    const pointSize = Math.max(1, Math.min(6, 50000 / data.length))
    
    data.forEach((point, index) => {
      const x = scaleX(point.x)
      const y = scaleY(point.y)
      
      // Skip points outside viewport for performance
      if (x < -10 || x > width + 10 || y < -10 || y > height + 10) return
      
      const color = point.color || colors.points[index % colors.points.length]
      context.fillStyle = color
      context.globalAlpha = Math.min(1, 50000 / data.length) // Fade for dense data
      
      context.beginPath()
      context.arc(x, y, pointSize, 0, Math.PI * 2)
      context.fill()
    })
    
    context.globalAlpha = 1
  }

  // Render line plot
  const renderLinePlot = (context, data, scaleX, scaleY, colors) => {
    if (data.length < 2) return
    
    // Sort data by x value for proper line connection
    const sortedData = [...data].sort((a, b) => a.x - b.x)
    
    context.strokeStyle = colors.points[0]
    context.lineWidth = 2
    context.beginPath()
    
    const startX = scaleX(sortedData[0].x)
    const startY = scaleY(sortedData[0].y)
    context.moveTo(startX, startY)
    
    for (let i = 1; i < sortedData.length; i++) {
      const x = scaleX(sortedData[i].x)
      const y = scaleY(sortedData[i].y)
      context.lineTo(x, y)
    }
    
    context.stroke()
    
    // Add points
    renderScatterPlot(context, sortedData, scaleX, scaleY, colors)
  }

  // Render heatmap
  const renderHeatmap = (context, data, scaleX, scaleY, colors) => {
    const gridSize = 20
    const heatmapData = new Map()
    
    // Aggregate data into grid
    data.forEach(point => {
      const gridX = Math.floor(scaleX(point.x) / gridSize)
      const gridY = Math.floor(scaleY(point.y) / gridSize)
      const key = `${gridX},${gridY}`
      
      heatmapData.set(key, (heatmapData.get(key) || 0) + 1)
    })
    
    const maxCount = Math.max(...heatmapData.values())
    
    // Render heatmap cells
    heatmapData.forEach((count, key) => {
      const [gridX, gridY] = key.split(',').map(Number)
      const intensity = count / maxCount
      const alpha = intensity * 0.8
      
      context.fillStyle = `rgba(59, 130, 246, ${alpha})`
      context.fillRect(gridX * gridSize, gridY * gridSize, gridSize, gridSize)
    })
  }

  // Render histogram
  const renderHistogram = (context, data, scaleX, scaleY, colors) => {
    const bins = 50
    const xMin = Math.min(...data.map(p => p.x))
    const xMax = Math.max(...data.map(p => p.x))
    const binWidth = (xMax - xMin) / bins
    
    const histogram = new Array(bins).fill(0)
    
    data.forEach(point => {
      const binIndex = Math.min(bins - 1, Math.floor((point.x - xMin) / binWidth))
      histogram[binIndex]++
    })
    
    const maxCount = Math.max(...histogram)
    const barWidth = (width - 80) / bins
    
    histogram.forEach((count, index) => {
      const x = 40 + index * barWidth
      const barHeight = (count / maxCount) * (height - 80)
      const y = height - 40 - barHeight
      
      context.fillStyle = colors.points[0]
      context.fillRect(x, y, barWidth - 1, barHeight)
    })
  }

  // Render labels and legends
  const renderLabels = (context, xMin, xMax, yMin, yMax, colors) => {
    context.fillStyle = colors.text
    context.font = '12px Arial'
    context.textAlign = 'center'
    
    // X axis labels
    for (let i = 0; i <= 5; i++) {
      const value = xMin + (xMax - xMin) * (i / 5)
      const x = 40 + (i / 5) * (width - 80)
      context.fillText(value.toFixed(1), x, height - 15)
    }
    
    // Y axis labels
    context.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const value = yMin + (yMax - yMin) * (i / 5)
      const y = height - 40 - (i / 5) * (height - 80)
      context.fillText(value.toFixed(1), 35, y + 4)
    }
  }

  // Mouse interaction handlers
  const handleMouseMove = throttle((event) => {
    if (!showTooltip || !interactive) return
    
    const rect = canvasRef.current.getBoundingClientRect()
    const x = (event.clientX - rect.left) * (width / rect.width)
    const y = (event.clientY - rect.top) * (height / rect.height)
    
    // Find nearest data point
    // This would need to be optimized for large datasets
    // Could use spatial indexing (quadtree, R-tree)
    
    setTooltip({
      visible: true,
      x: event.clientX,
      y: event.clientY,
      data: { x: x.toFixed(2), y: y.toFixed(2) }
    })
  }, 16) // ~60fps throttling

  const handleMouseLeave = () => {
    setTooltip({ visible: false, x: 0, y: 0, data: null })
  }

  // Zoom and pan handlers
  const handleWheel = useCallback((event) => {
    if (!interactive) return
    event.preventDefault()
    
    const delta = event.deltaY > 0 ? 0.9 : 1.1
    const newScale = Math.max(
      viewport.minScale,
      Math.min(viewport.maxScale, viewport.scale * delta)
    )
    
    setViewport(prev => ({ ...prev, scale: newScale }))
  }, [viewport, interactive])

  const resetZoom = () => {
    setViewport({ x: 0, y: 0, scale: 1, minScale: 0.1, maxScale: 10 })
  }

  const exportCanvas = () => {
    const canvas = canvasRef.current
    const link = document.createElement('a')
    link.download = `visualization-${Date.now()}.png`
    link.href = canvas.toDataURL()
    link.click()
  }

  return (
    <div className={`relative bg-white rounded-lg shadow-sm border ${className}`}>
      {/* Toolbar */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-4">
          <h3 className="text-lg font-semibold">Data Visualization</h3>
          <div className="text-sm text-gray-500">
            {renderStats.pointsRendered.toLocaleString()} pontos | 
            {renderStats.renderTime}ms | 
            {renderStats.fps} FPS
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={resetZoom}
            className="p-2 hover:bg-gray-100 rounded-md transition-colors"
            title="Reset Zoom"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setViewport(prev => ({ ...prev, scale: prev.scale * 1.2 }))}
            className="p-2 hover:bg-gray-100 rounded-md transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          
          <button
            onClick={() => setViewport(prev => ({ ...prev, scale: prev.scale * 0.8 }))}
            className="p-2 hover:bg-gray-100 rounded-md transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          
          <button
            onClick={exportCanvas}
            className="p-2 hover:bg-gray-100 rounded-md transition-colors"
            title="Export"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div className="relative">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
          className="cursor-crosshair"
        />
        
        {/* Loading overlay */}
        {isRendering && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span>Renderizando...</span>
            </div>
          </div>
        )}
        
        {/* Tooltip */}
        {tooltip.visible && (
          <div
            className="absolute z-10 bg-black text-white px-2 py-1 rounded text-sm pointer-events-none"
            style={{
              left: tooltip.x + 10,
              top: tooltip.y - 30,
              transform: 'translate(-50%, 0)'
            }}
          >
            X: {tooltip.data?.x}, Y: {tooltip.data?.y}
          </div>
        )}
      </div>
    </div>
  )
})

DataVisualization.displayName = 'DataVisualization'

export default DataVisualization