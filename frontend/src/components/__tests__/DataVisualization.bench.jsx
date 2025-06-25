import { bench, describe } from 'vitest'
import { render } from '@testing-library/react'
import DataVisualization from '../DataVisualization'

// Performance benchmarks for DataVisualization component
describe('DataVisualization Performance Benchmarks', () => {
  // Generate different dataset sizes for testing
  const generateTestData = (size) => {
    return Array.from({ length: size }, (_, i) => ({
      id: i,
      x: Math.random() * 1000,
      y: Math.random() * 1000,
      value: Math.random() * 100,
      category: `Category ${i % 10}`
    }))
  }

  const datasets = {
    small: generateTestData(1000),
    medium: generateTestData(10000),
    large: generateTestData(100000),
    massive: generateTestData(1000000)
  }

  describe('Rendering Performance', () => {
    bench('render with 1K data points', () => {
      render(
        <DataVisualization
          data={datasets.small}
          type="scatter"
          width={800}
          height={600}
          maxPoints={1000}
        />
      )
    })

    bench('render with 10K data points', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })

    bench('render with 100K data points (sampled)', () => {
      render(
        <DataVisualization
          data={datasets.large}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="intelligent"
        />
      )
    })

    bench('render with 1M data points (sampled)', () => {
      render(
        <DataVisualization
          data={datasets.massive}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="intelligent"
        />
      )
    })
  })

  describe('Sampling Performance', () => {
    bench('uniform sampling - 100K to 10K points', () => {
      render(
        <DataVisualization
          data={datasets.large}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="uniform"
        />
      )
    })

    bench('intelligent sampling - 100K to 10K points', () => {
      render(
        <DataVisualization
          data={datasets.large}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="intelligent"
        />
      )
    })

    bench('clustering sampling - 100K to 10K points', () => {
      render(
        <DataVisualization
          data={datasets.large}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="clustering"
        />
      )
    })
  })

  describe('Visualization Types Performance', () => {
    bench('scatter plot - 10K points', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })

    bench('line plot - 10K points', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="line"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })

    bench('heatmap - 10K points', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="heatmap"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })

    bench('histogram - 10K points', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="histogram"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })
  })

  describe('Canvas Rendering Performance', () => {
    bench('canvas rendering - high DPI', () => {
      // Mock high DPI scenario
      const originalDPR = window.devicePixelRatio
      window.devicePixelRatio = 2
      
      render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={1600}
          height={1200}
          maxPoints={10000}
        />
      )
      
      window.devicePixelRatio = originalDPR
    })

    bench('canvas rendering - multiple colors', () => {
      const coloredData = datasets.medium.map((point, index) => ({
        ...point,
        color: `hsl(${(index * 137.5) % 360}, 70%, 50%)`
      }))

      render(
        <DataVisualization
          data={coloredData}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })
  })

  describe('Interactive Features Performance', () => {
    bench('interactive mode enabled', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          interactive={true}
          showTooltip={true}
        />
      )
    })

    bench('interactive mode disabled', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          interactive={false}
          showTooltip={false}
        />
      )
    })
  })

  describe('Memory Usage Benchmarks', () => {
    bench('memory efficiency - large dataset', () => {
      // Test memory usage with large dataset
      const component = render(
        <DataVisualization
          data={datasets.large}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="intelligent"
        />
      )
      
      // Cleanup to test memory release
      component.unmount()
    })

    bench('memory efficiency - multiple renders', () => {
      // Test memory usage with multiple renders
      for (let i = 0; i < 5; i++) {
        const component = render(
          <DataVisualization
            data={datasets.medium}
            type="scatter"
            width={800}
            height={600}
            maxPoints={10000}
          />
        )
        component.unmount()
      }
    })
  })

  describe('Data Processing Benchmarks', () => {
    bench('data transformation - 100K points', () => {
      // Test data transformation performance
      const transformedData = datasets.large.map(point => ({
        ...point,
        normalizedX: point.x / 1000,
        normalizedY: point.y / 1000,
        category: point.category.toUpperCase()
      }))

      render(
        <DataVisualization
          data={transformedData}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })

    bench('data filtering - 100K points', () => {
      // Test data filtering performance
      const filteredData = datasets.large.filter(point => 
        point.x > 250 && point.x < 750 && point.y > 250 && point.y < 750
      )

      render(
        <DataVisualization
          data={filteredData}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )
    })
  })

  describe('Zoom and Pan Performance', () => {
    bench('viewport transformation - 10K points', () => {
      render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          interactive={true}
        />
      )
      
      // Simulate zoom operations
      // This would require triggering zoom events
    })
  })

  describe('Theme Switching Performance', () => {
    bench('theme switching - light to dark', () => {
      const { rerender } = render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          theme="light"
        />
      )

      rerender(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          theme="dark"
        />
      )
    })
  })

  describe('Export Performance', () => {
    bench('canvas export - 10K points', () => {
      const component = render(
        <DataVisualization
          data={datasets.medium}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )

      // Simulate export operation
      // This would require accessing the canvas element and calling toDataURL
      const canvas = component.container.querySelector('canvas')
      if (canvas) {
        canvas.toDataURL('image/png')
      }
    })
  })

  describe('Real-time Updates Performance', () => {
    bench('incremental data updates', () => {
      let currentData = datasets.small
      
      const { rerender } = render(
        <DataVisualization
          data={currentData}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
        />
      )

      // Simulate 10 incremental updates
      for (let i = 0; i < 10; i++) {
        const newPoint = {
          id: currentData.length + i,
          x: Math.random() * 1000,
          y: Math.random() * 1000,
          value: Math.random() * 100,
          category: `Category ${(currentData.length + i) % 10}`
        }
        
        currentData = [...currentData, newPoint]
        
        rerender(
          <DataVisualization
            data={currentData}
            type="scatter"
            width={800}
            height={600}
            maxPoints={10000}
          />
        )
      }
    })
  })

  describe('Web Worker Performance', () => {
    bench('web worker data processing', () => {
      // Test performance with web worker enabled
      render(
        <DataVisualization
          data={datasets.large}
          type="scatter"
          width={800}
          height={600}
          maxPoints={10000}
          sampling="intelligent"
        />
      )
      
      // This would test the web worker processing time
      // The actual timing would be handled by the web worker
    })
  })
})