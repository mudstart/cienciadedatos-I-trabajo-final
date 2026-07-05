-- =====================================================================
--  Esquema RDBMS del dataset OULAD (MySQL 8+)
--  Open University Learning Analytics Dataset — 7 tablas.
--  Reproduce el escenario sobre el que corre el pipeline (src/).
--  Carga: importar los CSV oficiales de OULAD o un volcado existente.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS oulad
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE oulad;

-- ----------------------------------------------------------------- courses
DROP TABLE IF EXISTS courses;
CREATE TABLE courses (
    code_module                VARCHAR(10),
    code_presentation          VARCHAR(10),
    module_presentation_length INT,
    PRIMARY KEY (code_module, code_presentation)
);

-- ------------------------------------------------------------- assessments
DROP TABLE IF EXISTS assessments;
CREATE TABLE assessments (
    id_assessment     INT PRIMARY KEY,
    code_module       VARCHAR(10),
    code_presentation VARCHAR(10),
    assessment_type   VARCHAR(10),
    date_due          INT,
    weight            FLOAT,
    KEY ix_a_curso (code_module, code_presentation)
);

-- ------------------------------------------------------------- student_info
DROP TABLE IF EXISTS student_info;
CREATE TABLE student_info (
    id_student           INT,
    code_module          VARCHAR(10),
    code_presentation    VARCHAR(10),
    gender               VARCHAR(5),
    region               VARCHAR(50),
    highest_education    VARCHAR(50),
    imd_band             VARCHAR(10),
    age_band             VARCHAR(10),
    num_of_prev_attempts TINYINT,
    studied_credits      SMALLINT,
    disability           VARCHAR(5),
    final_result         VARCHAR(15),    -- Pass | Fail | Withdrawn | Distinction
    PRIMARY KEY (id_student, code_module, code_presentation),
    KEY ix_si_result (final_result)
);

-- ------------------------------------------------------- student_registration
DROP TABLE IF EXISTS student_registration;
CREATE TABLE student_registration (
    id_student          INT,
    code_module         VARCHAR(10),
    code_presentation   VARCHAR(10),
    date_registration   INT,
    date_unregistration INT NULL,        -- no usar como feature (fuga -> Withdrawn)
    PRIMARY KEY (id_student, code_module, code_presentation)
);

-- -------------------------------------------------------- student_assessment
DROP TABLE IF EXISTS student_assessment;
CREATE TABLE student_assessment (
    id_assessment  INT,
    id_student     INT,
    date_submitted INT,
    is_banked      TINYINT,
    score          FLOAT,
    KEY ix_sa_assess (id_assessment),
    KEY ix_sa_student (id_student)
);

-- ----------------------------------------------------------------- vle
DROP TABLE IF EXISTS vle;
CREATE TABLE vle (
    id_site           INT PRIMARY KEY,
    code_module       VARCHAR(10),
    code_presentation VARCHAR(10),
    activity_type     VARCHAR(30),
    week_from         SMALLINT NULL,
    week_to           SMALLINT NULL
);

-- ----------------------------------------------------------- student_vle
DROP TABLE IF EXISTS student_vle;
CREATE TABLE student_vle (
    id_student        INT,
    id_site           INT,
    code_module       VARCHAR(10),
    code_presentation VARCHAR(10),
    date_interaction  INT,
    sum_click         INT,
    KEY ix_sv_student (id_student),
    KEY ix_sv_site (id_site)
);

-- =====================================================================
--  Vista analítica: tabla a nivel estudiante-módulo-presentación
--  enriquecida con la actividad del VLE (replica el scrub de Python).
-- =====================================================================
DROP VIEW IF EXISTS v_tabla_analitica;
CREATE VIEW v_tabla_analitica AS
SELECT
    si.id_student, si.code_module, si.code_presentation,
    si.gender, si.region, si.highest_education, si.imd_band, si.age_band,
    si.num_of_prev_attempts, si.studied_credits, si.disability,
    si.final_result,
    CASE WHEN si.final_result IN ('Pass','Distinction') THEN 1 ELSE 0 END AS target_bin,
    CASE si.final_result WHEN 'Withdrawn' THEN 0 WHEN 'Fail' THEN 1
         WHEN 'Pass' THEN 2 WHEN 'Distinction' THEN 3 END AS target_ord,
    r.date_registration,
    v.vle_total_clics, v.vle_interacciones, v.vle_sitios,
    a.score_medio
FROM student_info si
LEFT JOIN student_registration r
       ON r.id_student = si.id_student
      AND r.code_module = si.code_module
      AND r.code_presentation = si.code_presentation
LEFT JOIN (
    SELECT id_student, code_module, code_presentation,
           SUM(sum_click) vle_total_clics, COUNT(*) vle_interacciones,
           COUNT(DISTINCT id_site) vle_sitios
    FROM student_vle GROUP BY id_student, code_module, code_presentation
) v ON v.id_student = si.id_student AND v.code_module = si.code_module
   AND v.code_presentation = si.code_presentation
LEFT JOIN (
    SELECT sa.id_student, a.code_module, a.code_presentation,
           AVG(sa.score) score_medio
    FROM student_assessment sa
    JOIN assessments a ON a.id_assessment = sa.id_assessment
    GROUP BY sa.id_student, a.code_module, a.code_presentation
) a ON a.id_student = si.id_student AND a.code_module = si.code_module
   AND a.code_presentation = si.code_presentation;
