-- Create AI Project Database
CREATE DATABASE IF NOT EXISTS ai_project;
USE ai_project;

-- Create agent statistics table
CREATE TABLE IF NOT EXISTS agent_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(50) UNIQUE NOT NULL,
    total_episodes INT DEFAULT 0,
    best_reward FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_agent_name (agent_name),
    INDEX idx_best_reward (best_reward)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Insert initial data
INSERT IGNORE INTO agent_stats (agent_name, total_episodes, best_reward) VALUES
('cartpole', 0, 0),
('maze', 0, 0);

-- Create training history table (optional, for future analytics)
CREATE TABLE IF NOT EXISTS training_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    agent_name VARCHAR(50) NOT NULL,
    episodes INT,
    reward FLOAT,
    reset BOOLEAN DEFAULT FALSE,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_name) REFERENCES agent_stats(agent_name) ON DELETE CASCADE,
    INDEX idx_agent_name (agent_name),
    INDEX idx_trained_at (trained_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
