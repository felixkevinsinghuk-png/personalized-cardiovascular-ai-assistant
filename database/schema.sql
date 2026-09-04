-- schema.sql
-- Run this script once to set up the local MySQL database and all four tables.
-- Usage: mysql -u root -p < database/schema.sql

-- Create the database if it does not exist
CREATE DATABASE IF NOT EXISTS retinal_xai_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE retinal_xai_db;

-- ---------------------------------------------------------------------------
-- Table 1: predictions
-- Parent table — one row per uploaded retinal image.
-- All other tables reference this via prediction_id foreign key.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    image_filename  VARCHAR(255)    NOT NULL,
    dr_grade        TINYINT         NOT NULL COMMENT '0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative',
    dr_label        VARCHAR(50)     NOT NULL,
    dr_probability  FLOAT           NOT NULL COMMENT 'Softmax probability of predicted DR grade',
    cvd_score       FLOAT           NOT NULL COMMENT 'EfficientNet-B4 sigmoid output 0.0–1.0',
    fused_score     FLOAT           NOT NULL COMMENT 'Weighted fusion of DR and CVD scores',
    risk_level      ENUM('Low', 'Moderate', 'High') NOT NULL,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Table 2: heatmaps
-- Stores file paths of Grad-CAM heatmap images for each prediction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS heatmaps (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id   INT             NOT NULL,
    model_name      VARCHAR(50)     NOT NULL COMMENT 'e.g. resnet50 or efficientnet',
    heatmap_path    VARCHAR(512)    NOT NULL COMMENT 'Relative path under static/heatmaps/',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Table 3: llm_reports
-- Stores the full text of the LLM-generated cardiovascular risk report.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_reports (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id   INT             NOT NULL,
    llm_model       VARCHAR(100)    NOT NULL COMMENT 'e.g. biomistral or medgemma',
    report_text     LONGTEXT        NOT NULL,
    was_filtered    TINYINT(1)      NOT NULL DEFAULT 0 COMMENT '1 if safety filter was triggered',
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Table 4: chat_history
-- Stores every chatbot message exchange linked to a prediction.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_history (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id   INT             NOT NULL,
    user_message    TEXT            NOT NULL,
    llm_response    LONGTEXT        NOT NULL,
    was_filtered    TINYINT(1)      NOT NULL DEFAULT 0,
    created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
