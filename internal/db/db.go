package db

import (
	"database/sql"
	"fmt"
	"os"
	"time"

	_ "github.com/lib/pq"
)

var DB *sql.DB

// InitDB initializes the database connection
func InitDB() error {
	host := os.Getenv("DB_HOST")
	port := os.Getenv("DB_PORT")
	user := os.Getenv("DB_USER")
	password := os.Getenv("DB_PASSWORD")
	dbname := os.Getenv("DB_NAME")

	// If env vars are not set, try to use default values (development mode)
	if host == "" {
		host = "localhost"
	}
	if port == "" {
		port = "5432"
	}
	if user == "" {
		user = "lc_user"
	}
	if password == "" {
		password = "lc_password"
	}
	if dbname == "" {
		dbname = "lc_db"
	}

	connStr := fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		host, port, user, password, dbname)

	var err error
	DB, err = sql.Open("postgres", connStr)
	if err != nil {
		return fmt.Errorf("failed to open database connection: %w", err)
	}

	// Retry connection
	for i := 0; i < 10; i++ {
		err = DB.Ping()
		if err == nil {
			break
		}
		time.Sleep(1 * time.Second)
	}
	if err != nil {
		return fmt.Errorf("failed to ping database: %w", err)
	}

	return createSchema()
}

func createSchema() error {
	query := `
	CREATE TABLE IF NOT EXISTS solved_problems (
		id SERIAL PRIMARY KEY,
		problem_id INT NOT NULL,
		timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		UNIQUE(problem_id)
	);
	CREATE TABLE IF NOT EXISTS problem_code (
		problem_id INT PRIMARY KEY,
		code TEXT NOT NULL,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	);
	`
	_, err := DB.Exec(query)
	return err
}

// MarkProblemSolved marks a problem as solved
func MarkProblemSolved(problemID int) error {
	query := `
	INSERT INTO solved_problems (problem_id) 
	VALUES ($1) 
	ON CONFLICT (problem_id) DO NOTHING
	`
	_, err := DB.Exec(query, problemID)
	return err
}

// IsProblemSolved checks if a problem is solved
func IsProblemSolved(problemID int) (bool, error) {
	var exists bool
	query := `SELECT EXISTS(SELECT 1 FROM solved_problems WHERE problem_id = $1)`
	err := DB.QueryRow(query, problemID).Scan(&exists)
	return exists, err
}

// GetSolvedProblems returns a map of solved problem IDs
func GetSolvedProblems() (map[int]bool, error) {
	rows, err := DB.Query("SELECT problem_id FROM solved_problems")
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	solved := make(map[int]bool)
	for rows.Next() {
		var id int
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		solved[id] = true
	}
	return solved, nil
}

// SaveCode saves the code for a problem
func SaveCode(problemID int, code string) error {
	query := `
	INSERT INTO problem_code (problem_id, code, updated_at)
	VALUES (, , CURRENT_TIMESTAMP)
	ON CONFLICT (problem_id) DO UPDATE
	SET code = EXCLUDED.code, updated_at = CURRENT_TIMESTAMP
	`
	_, err := DB.Exec(query, problemID, code)
	return err
}

// GetCode retrieves the saved code for a problem
func GetCode(problemID int) (string, error) {
	var code string
	query := `SELECT code FROM problem_code WHERE problem_id = `
	err := DB.QueryRow(query, problemID).Scan(&code)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return code, err
}
