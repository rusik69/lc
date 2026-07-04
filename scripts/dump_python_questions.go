// Command: dump_python_questions
// Build: go run scripts/dump_python_questions.go
// Output: prints JSON manifest of all Python MC questions and their correct answers
package main

import (
	"encoding/json"
	"fmt"
	"os"

	_ "github.com/rusik69/lc/internal/courses/python"
	"github.com/rusik69/lc/internal/problems"
)

type QuestionEntry struct {
	ID            int      `json:"id"`
	Text          string   `json:"text"`
	Options       []string `json:"options"`
	CorrectIndex  int      `json:"correct_index"`
	CorrectAnswer string   `json:"correct_answer"`
	Explanation   string   `json:"explanation"`
}

func main() {
	questions := problems.GetPythonQuestions()
	entries := make([]QuestionEntry, 0, len(questions))

	issues := 0
	for _, q := range questions {
		entry := QuestionEntry{
			ID:            q.ID,
			Text:          q.Text,
			Options:       q.Options[:],
			CorrectIndex:  q.CorrectAnswer,
			CorrectAnswer: q.Options[q.CorrectAnswer],
			Explanation:   q.Explanation,
		}
		entries = append(entries, entry)

		// Check for suspicious patterns
		if q.CorrectAnswer < 0 || q.CorrectAnswer > 3 {
			fmt.Fprintf(os.Stderr, "WARN: Q%d CorrectAnswer out of range: %d\n", q.ID, q.CorrectAnswer)
			issues++
		}
		if q.Options[q.CorrectAnswer] == "" {
			fmt.Fprintf(os.Stderr, "WARN: Q%d correct answer is empty string for index %d\n", q.ID, q.CorrectAnswer)
			issues++
		}
	}

	jsonData, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error marshaling JSON: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile("/home/ubuntu/lc/python_questions_manifest.json", jsonData, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing file: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Wrote %d Python questions to python_questions_manifest.json\n", len(entries))
	if issues > 0 {
		fmt.Fprintf(os.Stderr, "Found %d issues\n", issues)
		os.Exit(1)
	}
}
