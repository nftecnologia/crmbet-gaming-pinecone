/**
 * User Service Tests
 * 
 * Testes unitários completos para userService
 * com mocks e cenários realistas
 * 
 * @author CRM Team
 */

const userService = require('../../src/services/userService');
const userRepository = require('../../src/repositories/userRepository');
const { cache } = require('../../src/config/redis');

// Mocks
jest.mock('../../src/repositories/userRepository');
jest.mock('../../src/config/redis');
jest.mock('../../src/utils/logger', () => ({
  error: jest.fn(),
  debug: jest.fn(),
  business: jest.fn(),
  info: jest.fn()
}));

describe('UserService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('getUserSegment', () => {
    const mockUser = {
      id: 1,
      external_id: 'user_123',
      email: 'test@example.com',
      name: 'Test User',
      segment: 'medium_value',
      cluster_id: 5,
      registration_date: '2023-01-01T00:00:00Z',
      last_activity: '2024-01-10T12:00:00Z',
      total_deposits: 500,
      total_withdrawals: 100,
      total_bets: 1000,
      bet_count: 50,
      win_rate: 65.5
    };

    it('should return user segment data for existing user', async () => {
      // Arrange
      userRepository.getUserById.mockResolvedValue(mockUser);
      userRepository.updateUserSegment.mockResolvedValue(mockUser);

      // Act
      const result = await userService.getUserSegment(1);

      // Assert
      expect(result).toBeDefined();
      expect(result.user_id).toBe(1);
      expect(result.segment).toBeDefined();
      expect(result.confidence_score).toBeGreaterThan(0);
      expect(result.segment_features).toBeDefined();
      expect(result.recommendations).toBeDefined();
      expect(Array.isArray(result.recommendations)).toBe(true);
    });

    it('should return null for non-existent user', async () => {
      // Arrange
      userRepository.getUserById.mockResolvedValue(null);

      // Act
      const result = await userService.getUserSegment(999);

      // Assert
      expect(result).toBeNull();
    });

    it('should classify high-value user correctly', async () => {
      // Arrange
      const highValueUser = {
        ...mockUser,
        total_deposits: 2000,
        total_withdrawals: 200,
        total_bets: 5000,
        bet_count: 200,
        last_activity: new Date().toISOString()
      };
      
      userRepository.getUserById.mockResolvedValue(highValueUser);
      userRepository.updateUserSegment.mockResolvedValue(highValueUser);

      // Act
      const result = await userService.getUserSegment(1);

      // Assert
      expect(result.segment).toBe('high_value');
      expect(result.confidence_score).toBeGreaterThan(0.8);
      expect(result.recommendations).toContainEqual(
        expect.objectContaining({
          type: 'campaign',
          title: 'VIP Exclusive Offer'
        })
      );
    });

    it('should classify new user correctly', async () => {
      // Arrange
      const newUser = {
        ...mockUser,
        registration_date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days ago
        total_deposits: 50,
        bet_count: 2
      };
      
      userRepository.getUserById.mockResolvedValue(newUser);
      userRepository.updateUserSegment.mockResolvedValue({ ...newUser, segment: 'new_user' });

      // Act
      const result = await userService.getUserSegment(1);

      // Assert
      expect(result.segment).toBe('new_user');
      expect(result.recommendations).toContainEqual(
        expect.objectContaining({
          type: 'onboarding',
          title: 'Welcome Journey'
        })
      );
    });

    it('should classify inactive user correctly', async () => {
      // Arrange
      const inactiveUser = {
        ...mockUser,
        last_activity: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000).toISOString() // 45 days ago
      };
      
      userRepository.getUserById.mockResolvedValue(inactiveUser);
      userRepository.updateUserSegment.mockResolvedValue({ ...inactiveUser, segment: 'inactive' });

      // Act
      const result = await userService.getUserSegment(1);

      // Assert
      expect(result.segment).toBe('inactive');
      expect(result.recommendations.some(r => r.type === 'retention')).toBe(true);
    });

    it('should handle repository errors gracefully', async () => {
      // Arrange
      userRepository.getUserById.mockRejectedValue(new Error('Database error'));

      // Act & Assert
      await expect(userService.getUserSegment(1)).rejects.toThrow('Database error');
    });
  });

  describe('getSegmentStats', () => {
    const mockStats = {
      total_users: 1000,
      segments: {
        high_value: { count: 50, percentage: 5.0, avg_ltv: 2500 },
        medium_value: { count: 200, percentage: 20.0, avg_ltv: 800 },
        low_value: { count: 400, percentage: 40.0, avg_ltv: 200 },
        new_user: { count: 250, percentage: 25.0, avg_ltv: 50 },
        inactive: { count: 100, percentage: 10.0, avg_ltv: 0 }
      }
    };

    it('should return segment statistics', async () => {
      // Arrange
      userRepository.getSegmentStatistics.mockResolvedValue(mockStats);

      // Act
      const result = await userService.getSegmentStats();

      // Assert
      expect(result).toBeDefined();
      expect(result.total_users).toBe(1000);
      expect(result.segments).toBeDefined();
      expect(result.segments.high_value.count).toBe(50);
      expect(result.segments.high_value.percentage).toBe(5.0);
    });

    it('should handle empty statistics', async () => {
      // Arrange
      const emptyStats = {
        total_users: 0,
        segments: {
          high_value: { count: 0, percentage: 0, avg_ltv: 0 },
          medium_value: { count: 0, percentage: 0, avg_ltv: 0 },
          low_value: { count: 0, percentage: 0, avg_ltv: 0 },
          new_user: { count: 0, percentage: 0, avg_ltv: 0 },
          inactive: { count: 0, percentage: 0, avg_ltv: 0 }
        }
      };
      
      userRepository.getSegmentStatistics.mockResolvedValue(emptyStats);

      // Act
      const result = await userService.getSegmentStats();

      // Assert
      expect(result.total_users).toBe(0);
      expect(Object.values(result.segments).every(s => s.count === 0)).toBe(true);
    });
  });

  describe('getUserBehaviorAnalysis', () => {
    const mockUser = {
      id: 1,
      registration_date: '2023-01-01T00:00:00Z',
      last_activity: '2024-01-10T12:00:00Z',
      total_deposits: 500,
      total_withdrawals: 100,
      total_bets: 1000,
      bet_count: 50,
      win_rate: 65.5
    };

    const mockBehaviorData = {
      activity_level: 80,
      engagement_level: 'high'
    };

    it('should return behavior analysis for valid user', async () => {
      // Arrange
      userRepository.getUserById.mockResolvedValue(mockUser);
      userRepository.getUserBehaviorData.mockResolvedValue(mockBehaviorData);

      // Act
      const result = await userService.getUserBehaviorAnalysis(1, '30d');

      // Assert
      expect(result).toBeDefined();
      expect(result.user_id).toBe(1);
      expect(result.period).toBe('30d');
      expect(result.behavior_metrics).toBeDefined();
      expect(result.behavior_metrics.activity_score).toBeDefined();
      expect(result.behavior_metrics.engagement_score).toBeDefined();
      expect(result.patterns).toBeDefined();
      expect(result.predictions).toBeDefined();
    });

    it('should handle different time periods', async () => {
      // Arrange
      userRepository.getUserById.mockResolvedValue(mockUser);
      userRepository.getUserBehaviorData.mockResolvedValue(mockBehaviorData);

      // Act
      const result7d = await userService.getUserBehaviorAnalysis(1, '7d');
      const result90d = await userService.getUserBehaviorAnalysis(1, '90d');

      // Assert
      expect(result7d.period).toBe('7d');
      expect(result90d.period).toBe('90d');
    });

    it('should return null for non-existent user', async () => {
      // Arrange
      userRepository.getUserById.mockResolvedValue(null);

      // Act
      const result = await userService.getUserBehaviorAnalysis(999);

      // Assert
      expect(result).toBeNull();
    });
  });

  describe('searchUsers', () => {
    const mockSearchResult = {
      users: [
        { id: 1, email: 'user1@example.com', segment: 'high_value' },
        { id: 2, email: 'user2@example.com', segment: 'medium_value' }
      ],
      pagination: {
        page: 1,
        limit: 20,
        total: 2,
        pages: 1
      }
    };

    it('should search users with criteria', async () => {
      // Arrange
      const criteria = { segment: 'high_value', page: 1, limit: 20 };
      userRepository.searchUsers.mockResolvedValue(mockSearchResult);

      // Act
      const result = await userService.searchUsers(criteria);

      // Assert
      expect(result).toBeDefined();
      expect(result.users).toHaveLength(2);
      expect(result.pagination).toBeDefined();
      expect(userRepository.searchUsers).toHaveBeenCalledWith(criteria);
    });

    it('should handle empty search results', async () => {
      // Arrange
      const emptyResult = {
        users: [],
        pagination: { page: 1, limit: 20, total: 0, pages: 0 }
      };
      userRepository.searchUsers.mockResolvedValue(emptyResult);

      // Act
      const result = await userService.searchUsers({ email: 'nonexistent@example.com' });

      // Assert
      expect(result.users).toHaveLength(0);
      expect(result.pagination.total).toBe(0);
    });
  });

  describe('getUserRecommendations', () => {
    const mockSegmentData = {
      user_id: 1,
      segment: 'high_value',
      segment_features: {
        ltv: 400,
        activity_score: 85,
        churn_risk: 0.2
      },
      recommendations: [
        { type: 'campaign', title: 'VIP Exclusive Offer', priority: 'high' }
      ]
    };

    it('should return enhanced recommendations', async () => {
      // Arrange
      jest.spyOn(userService, 'getUserSegment').mockResolvedValue(mockSegmentData);

      // Act
      const result = await userService.getUserRecommendations(1);

      // Assert
      expect(result).toBeDefined();
      expect(result.user_id).toBe(1);
      expect(result.recommendations).toBeDefined();
      expect(Array.isArray(result.recommendations)).toBe(true);
      
      // Check enhanced recommendations have additional fields
      if (result.recommendations.length > 0) {
        const rec = result.recommendations[0];
        expect(rec.id).toBeDefined();
        expect(rec.confidence).toBeDefined();
        expect(rec.expected_impact).toBeDefined();
        expect(rec.valid_until).toBeDefined();
      }
    });

    it('should return null for non-existent user', async () => {
      // Arrange
      jest.spyOn(userService, 'getUserSegment').mockResolvedValue(null);

      // Act
      const result = await userService.getUserRecommendations(999);

      // Assert
      expect(result).toBeNull();
    });
  });

  describe('UserSegmentationEngine', () => {
    const { UserSegmentationEngine } = userService;
    let engine;

    beforeEach(() => {
      engine = new UserSegmentationEngine();
    });

    describe('calculateBehaviorFeatures', () => {
      it('should calculate features correctly', () => {
        // Arrange
        const userData = {
          registration_date: new Date(Date.now() - 100 * 24 * 60 * 60 * 1000).toISOString(), // 100 days ago
          last_activity: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(), // 5 days ago
          total_deposits: 1000,
          total_withdrawals: 200,
          total_bets: 2000,
          bet_count: 100,
          win_rate: 60
        };

        // Act
        const features = engine.calculateBehaviorFeatures(userData);

        // Assert
        expect(features.ltv).toBe(800); // 1000 - 200
        expect(features.bet_frequency).toBeCloseTo(1); // 100 bets / 100 days
        expect(features.avg_bet_size).toBe(20); // 2000 / 100
        expect(features.days_since_last_activity).toBe(5);
        expect(features.registration_days).toBe(100);
        expect(features.activity_score).toBeGreaterThan(0);
        expect(features.churn_risk).toBeGreaterThanOrEqual(0);
        expect(features.churn_risk).toBeLessThanOrEqual(1);
      });

      it('should handle null last activity', () => {
        // Arrange
        const userData = {
          registration_date: new Date().toISOString(),
          last_activity: null,
          total_deposits: 100,
          total_withdrawals: 0,
          total_bets: 200,
          bet_count: 10,
          win_rate: 50
        };

        // Act
        const features = engine.calculateBehaviorFeatures(userData);

        // Assert
        expect(features.days_since_last_activity).toBe(999);
        expect(features.activity_score).toBe(0);
      });
    });

    describe('calculateChurnRisk', () => {
      it('should calculate low churn risk for active users', () => {
        // Arrange
        const features = {
          daysSinceLastActivity: 1,
          betFrequency: 5,
          ltv: 1000,
          winRate: 70
        };

        // Act
        const churnRisk = engine.calculateChurnRisk(features);

        // Assert
        expect(churnRisk).toBeLessThan(0.5);
      });

      it('should calculate high churn risk for inactive users', () => {
        // Arrange
        const features = {
          daysSinceLastActivity: 30,
          betFrequency: 0.1,
          ltv: 10,
          winRate: 20
        };

        // Act
        const churnRisk = engine.calculateChurnRisk(features);

        // Assert
        expect(churnRisk).toBeGreaterThan(0.5);
      });
    });

    describe('determineSegment', () => {
      it('should classify high value user', () => {
        // Arrange
        const features = {
          total_deposits: 2000,
          ltv: 3000,
          bet_frequency: 10,
          registration_days: 200,
          days_since_last_activity: 1,
          activity_score: 95,
          churn_risk: 0.1
        };

        // Act
        const result = engine.determineSegment(features);

        // Assert
        expect(result.segment).toBe('high_value');
        expect(result.confidence).toBeGreaterThan(0.9);
      });

      it('should classify inactive user', () => {
        // Arrange
        const features = {
          total_deposits: 500,
          ltv: 300,
          bet_frequency: 2,
          registration_days: 100,
          days_since_last_activity: 45,
          activity_score: 10,
          churn_risk: 0.8
        };

        // Act
        const result = engine.determineSegment(features);

        // Assert
        expect(result.segment).toBe('inactive');
        expect(result.confidence).toBeGreaterThan(0.8);
      });

      it('should classify new user', () => {
        // Arrange
        const features = {
          total_deposits: 50,
          ltv: 30,
          bet_frequency: 1,
          registration_days: 10,
          days_since_last_activity: 1,
          activity_score: 80,
          churn_risk: 0.3
        };

        // Act
        const result = engine.determineSegment(features);

        // Assert
        expect(result.segment).toBe('new_user');
        expect(result.confidence).toBeGreaterThan(0.8);
      });
    });

    describe('generateRecommendations', () => {
      it('should generate high value recommendations', () => {
        // Arrange
        const features = { ltv: 3000, churn_risk: 0.1 };

        // Act
        const recommendations = engine.generateRecommendations('high_value', features);

        // Assert
        expect(recommendations).toContainEqual(
          expect.objectContaining({
            type: 'campaign',
            title: 'VIP Exclusive Offer',
            priority: 'high'
          })
        );
      });

      it('should generate new user recommendations', () => {
        // Arrange
        const features = { ltv: 50, churn_risk: 0.3 };

        // Act
        const recommendations = engine.generateRecommendations('new_user', features);

        // Assert
        expect(recommendations).toContainEqual(
          expect.objectContaining({
            type: 'onboarding',
            title: 'Welcome Journey',
            priority: 'high'
          })
        );
      });

      it('should generate retention recommendations for high-risk inactive users', () => {
        // Arrange
        const features = { ltv: 200, churn_risk: 0.8 };

        // Act
        const recommendations = engine.generateRecommendations('inactive', features);

        // Assert
        expect(recommendations).toContainEqual(
          expect.objectContaining({
            type: 'retention',
            title: 'Win-back Campaign',
            priority: 'high'
          })
        );
      });
    });
  });
});