import { create } from 'zustand'
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'

// Auth store
export const useAuthStore = create()(
  devtools(
    persist(
      (set, get) => ({
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,

        login: (user, token) => {
          set((state) => {
            state.user = user
            state.token = token
            state.isAuthenticated = true
          })
        },

        logout: () => {
          set((state) => {
            state.user = null
            state.token = null
            state.isAuthenticated = false
          })
          localStorage.removeItem('authToken')
        },

        updateUser: (userData) => {
          set((state) => {
            if (state.user) {
              state.user = { ...state.user, ...userData }
            }
          })
        },

        setLoading: (loading) => {
          set((state) => {
            state.isLoading = loading
          })
        },
      }),
      {
        name: 'auth-store',
        partialize: (state) => ({
          user: state.user,
          token: state.token,
          isAuthenticated: state.isAuthenticated,
        }),
      }
    ),
    { name: 'auth-store' }
  )
)

// App settings store
export const useSettingsStore = create()(
  devtools(
    persist(
      immer((set, get) => ({
        theme: 'light',
        sidebarCollapsed: false,
        language: 'pt-BR',
        dateFormat: 'DD/MM/YYYY',
        timezone: 'America/Sao_Paulo',
        notifications: {
          email: true,
          push: true,
          browser: true,
          campaigns: true,
          clusters: true,
          users: true,
        },
        dashboard: {
          refreshInterval: 30000, // 30 seconds
          autoRefresh: true,
          defaultTimeRange: '30d',
        },

        setTheme: (theme) => {
          set((state) => {
            state.theme = theme
          })
        },

        setSidebarCollapsed: (collapsed) => {
          set((state) => {
            state.sidebarCollapsed = collapsed
          })
        },

        setLanguage: (language) => {
          set((state) => {
            state.language = language
          })
        },

        updateNotifications: (notifications) => {
          set((state) => {
            state.notifications = { ...state.notifications, ...notifications }
          })
        },

        updateDashboard: (dashboard) => {
          set((state) => {
            state.dashboard = { ...state.dashboard, ...dashboard }
          })
        },

        resetSettings: () => {
          set((state) => {
            state.theme = 'light'
            state.sidebarCollapsed = false
            state.language = 'pt-BR'
            state.dateFormat = 'DD/MM/YYYY'
            state.timezone = 'America/Sao_Paulo'
            state.notifications = {
              email: true,
              push: true,
              browser: true,
              campaigns: true,
              clusters: true,
              users: true,
            }
            state.dashboard = {
              refreshInterval: 30000,
              autoRefresh: true,
              defaultTimeRange: '30d',
            }
          })
        },
      })),
      {
        name: 'settings-store',
      }
    ),
    { name: 'settings-store' }
  )
)

// Notifications store
export const useNotificationsStore = create()(
  devtools(
    subscribeWithSelector(
      immer((set, get) => ({
        notifications: [],
        unreadCount: 0,
        loading: false,
        lastFetch: null,

        addNotification: (notification) => {
          set((state) => {
            const newNotification = {
              id: Date.now().toString(),
              timestamp: new Date().toISOString(),
              read: false,
              ...notification,
            }
            state.notifications.unshift(newNotification)
            if (!newNotification.read) {
              state.unreadCount += 1
            }
          })
        },

        removeNotification: (id) => {
          set((state) => {
            const index = state.notifications.findIndex(n => n.id === id)
            if (index !== -1) {
              if (!state.notifications[index].read) {
                state.unreadCount -= 1
              }
              state.notifications.splice(index, 1)
            }
          })
        },

        markAsRead: (id) => {
          set((state) => {
            const notification = state.notifications.find(n => n.id === id)
            if (notification && !notification.read) {
              notification.read = true
              state.unreadCount -= 1
            }
          })
        },

        markAllAsRead: () => {
          set((state) => {
            state.notifications.forEach(n => {
              if (!n.read) {
                n.read = true
              }
            })
            state.unreadCount = 0
          })
        },

        setNotifications: (notifications) => {
          set((state) => {
            state.notifications = notifications
            state.unreadCount = notifications.filter(n => !n.read).length
            state.lastFetch = new Date().toISOString()
          })
        },

        setLoading: (loading) => {
          set((state) => {
            state.loading = loading
          })
        },

        clearAll: () => {
          set((state) => {
            state.notifications = []
            state.unreadCount = 0
          })
        },
      }))
    ),
    { name: 'notifications-store' }
  )
)

// Dashboard data store
export const useDashboardStore = create()(
  devtools(
    immer((set, get) => ({
      metrics: null,
      charts: {
        revenue: null,
        users: null,
        conversion: null,
        clusters: null,
      },
      activities: [],
      insights: [],
      loading: false,
      lastUpdate: null,
      timeRange: '30d',

      setMetrics: (metrics) => {
        set((state) => {
          state.metrics = metrics
          state.lastUpdate = new Date().toISOString()
        })
      },

      setChartData: (chartType, data) => {
        set((state) => {
          state.charts[chartType] = data
        })
      },

      setActivities: (activities) => {
        set((state) => {
          state.activities = activities
        })
      },

      addActivity: (activity) => {
        set((state) => {
          state.activities.unshift({
            id: Date.now().toString(),
            timestamp: new Date().toISOString(),
            ...activity,
          })
          // Keep only the last 50 activities
          if (state.activities.length > 50) {
            state.activities = state.activities.slice(0, 50)
          }
        })
      },

      setInsights: (insights) => {
        set((state) => {
          state.insights = insights
        })
      },

      setTimeRange: (timeRange) => {
        set((state) => {
          state.timeRange = timeRange
        })
      },

      setLoading: (loading) => {
        set((state) => {
          state.loading = loading
        })
      },

      refreshData: async () => {
        // This would trigger data refresh from components
        set((state) => {
          state.lastUpdate = new Date().toISOString()
        })
      },
    })),
    { name: 'dashboard-store' }
  )
)

// Filters and search store
export const useFiltersStore = create()(
  devtools(
    immer((set, get) => ({
      globalSearch: '',
      users: {
        search: '',
        cluster: '',
        status: 'all',
        dateRange: null,
        sortBy: 'createdAt',
        sortOrder: 'desc',
      },
      campaigns: {
        search: '',
        status: 'all',
        type: 'all',
        cluster: '',
        dateRange: null,
        sortBy: 'createdAt',
        sortOrder: 'desc',
      },
      clusters: {
        search: '',
        status: 'all',
        sortBy: 'users',
        sortOrder: 'desc',
      },

      setGlobalSearch: (search) => {
        set((state) => {
          state.globalSearch = search
        })
      },

      setUserFilters: (filters) => {
        set((state) => {
          state.users = { ...state.users, ...filters }
        })
      },

      setCampaignFilters: (filters) => {
        set((state) => {
          state.campaigns = { ...state.campaigns, ...filters }
        })
      },

      setClusterFilters: (filters) => {
        set((state) => {
          state.clusters = { ...state.clusters, ...filters }
        })
      },

      resetFilters: (section) => {
        set((state) => {
          if (section === 'users') {
            state.users = {
              search: '',
              cluster: '',
              status: 'all',
              dateRange: null,
              sortBy: 'createdAt',
              sortOrder: 'desc',
            }
          } else if (section === 'campaigns') {
            state.campaigns = {
              search: '',
              status: 'all',
              type: 'all',
              cluster: '',
              dateRange: null,
              sortBy: 'createdAt',
              sortOrder: 'desc',
            }
          } else if (section === 'clusters') {
            state.clusters = {
              search: '',
              status: 'all',
              sortBy: 'users',
              sortOrder: 'desc',
            }
          }
        })
      },

      resetAllFilters: () => {
        set((state) => {
          state.globalSearch = ''
          state.users = {
            search: '',
            cluster: '',
            status: 'all',
            dateRange: null,
            sortBy: 'createdAt',
            sortOrder: 'desc',
          }
          state.campaigns = {
            search: '',
            status: 'all',
            type: 'all',
            cluster: '',
            dateRange: null,
            sortBy: 'createdAt',
            sortOrder: 'desc',
          }
          state.clusters = {
            search: '',
            status: 'all',
            sortBy: 'users',
            sortOrder: 'desc',
          }
        })
      },
    })),
    { name: 'filters-store' }
  )
)

// UI state store
export const useUIStore = create()(
  devtools(
    immer((set, get) => ({
      modals: {
        createCampaign: false,
        editCampaign: null,
        createCluster: false,
        editCluster: null,
        userDetails: null,
      },
      loading: {
        global: false,
        users: false,
        campaigns: false,
        clusters: false,
        analytics: false,
      },
      errors: {},
      toasts: [],

      openModal: (modal, data = null) => {
        set((state) => {
          state.modals[modal] = data || true
        })
      },

      closeModal: (modal) => {
        set((state) => {
          state.modals[modal] = false
        })
      },

      closeAllModals: () => {
        set((state) => {
          Object.keys(state.modals).forEach(key => {
            state.modals[key] = false
          })
        })
      },

      setLoading: (section, loading) => {
        set((state) => {
          state.loading[section] = loading
        })
      },

      setError: (section, error) => {
        set((state) => {
          state.errors[section] = error
        })
      },

      clearError: (section) => {
        set((state) => {
          delete state.errors[section]
        })
      },

      clearAllErrors: () => {
        set((state) => {
          state.errors = {}
        })
      },

      addToast: (toast) => {
        set((state) => {
          const newToast = {
            id: Date.now().toString(),
            ...toast,
          }
          state.toasts.push(newToast)
        })
      },

      removeToast: (id) => {
        set((state) => {
          state.toasts = state.toasts.filter(t => t.id !== id)
        })
      },
    })),
    { name: 'ui-store' }
  )
)

// Combined store selector helpers
export const useStores = () => ({
  auth: useAuthStore(),
  settings: useSettingsStore(),
  notifications: useNotificationsStore(),
  dashboard: useDashboardStore(),
  filters: useFiltersStore(),
  ui: useUIStore(),
})

// Utility to reset all stores (useful for logout)
export const resetAllStores = () => {
  useAuthStore.getState().logout()
  useSettingsStore.getState().resetSettings()
  useNotificationsStore.getState().clearAll()
  useFiltersStore.getState().resetAllFilters()
  useUIStore.getState().closeAllModals()
  useUIStore.getState().clearAllErrors()
}