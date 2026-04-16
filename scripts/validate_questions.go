package main

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/rusik69/lc/internal/problems"
	// Import all course packages to trigger init() functions
	_ "github.com/rusik69/lc/internal/courses/algorithms"
	_ "github.com/rusik69/lc/internal/courses/aws"
	_ "github.com/rusik69/lc/internal/courses/azure"
	_ "github.com/rusik69/lc/internal/courses/computer_architecture"
	_ "github.com/rusik69/lc/internal/courses/devops"
	_ "github.com/rusik69/lc/internal/courses/frontend"
	_ "github.com/rusik69/lc/internal/courses/golang"
	_ "github.com/rusik69/lc/internal/courses/kubernetes"
	_ "github.com/rusik69/lc/internal/courses/linux"
	_ "github.com/rusik69/lc/internal/courses/machine_learning"
	_ "github.com/rusik69/lc/internal/courses/math"
	_ "github.com/rusik69/lc/internal/courses/networking"
	_ "github.com/rusik69/lc/internal/courses/python"
	_ "github.com/rusik69/lc/internal/courses/software_architecture"
	_ "github.com/rusik69/lc/internal/courses/systems_design"
)

// ValidationError represents a validation error for a question
type ValidationError struct {
	Topic      string
	QuestionID int
	ErrorType  string
	Message    string
}

// brokenSplitRe matches the residue of an older splitter that broke
// explanations on every "." — e.g. "(127. Step 2: 0." or "e. Step 3: g.".
var brokenSplitRe = regexp.MustCompile(`\b[A-Za-z0-9]\. Step \d+:`)

// stepScaffoldRe matches the "Step N:" scaffolding that predates the
// switch to flowing-prose explanations.
var stepScaffoldRe = regexp.MustCompile(`(?i)\bStep \d+:`)

// TopicStats represents statistics for a topic
type TopicStats struct {
	Topic             string
	TotalQuestions    int
	Errors            int
	AllOfAboveCount   int
	AllOfAbovePercent float64
	DuplicateCount    int
}

func main() {
	topics := []string{
		"algorithms",
		"aws",
		"azure",
		"computer_architecture",
		"devops",
		"frontend",
		"golang",
		"kubernetes",
		"linux",
		"machine_learning",
		"math",
		"networking",
		"python",
		"software_architecture",
		"systems_design",
	}

	allErrors := []ValidationError{}
	allStats := []TopicStats{}

	fmt.Println("=== Question Validation Report ===")
	fmt.Println()

	for _, topic := range topics {
		errors, stats := validateTopic(topic)
		allErrors = append(allErrors, errors...)
		allStats = append(allStats, stats)
	}

	// Print statistics summary
	fmt.Println("\n=== Summary Statistics ===\n")
	fmt.Printf("%-25s | %8s | %8s | %15s | %12s\n", "Topic", "Questions", "Errors", "All of Above", "Duplicates")
	fmt.Println(strings.Repeat("-", 90))

	totalQuestions := 0
	totalErrors := 0
	for _, stats := range allStats {
		totalQuestions += stats.TotalQuestions
		totalErrors += stats.Errors
		fmt.Printf("%-25s | %8d | %8d | %8d (%.1f%%) | %12d\n",
			stats.Topic,
			stats.TotalQuestions,
			stats.Errors,
			stats.AllOfAboveCount,
			stats.AllOfAbovePercent,
			stats.DuplicateCount,
		)
	}
	fmt.Println(strings.Repeat("-", 90))
	fmt.Printf("%-25s | %8d | %8d |\n", "TOTAL", totalQuestions, totalErrors)

	// Print detailed errors
	if len(allErrors) > 0 {
		fmt.Println("\n=== Detailed Errors ===\n")
		for _, err := range allErrors {
			fmt.Printf("[%s] ID %d - %s: %s\n",
				err.Topic,
				err.QuestionID,
				err.ErrorType,
				err.Message,
			)
		}
		fmt.Printf("\nTotal errors found: %d\n", len(allErrors))
		os.Exit(1)
	} else {
		fmt.Println("\n✓ All questions validated successfully!")
		os.Exit(0)
	}
}

func validateTopic(topic string) ([]ValidationError, TopicStats) {
	errors := []ValidationError{}
	stats := TopicStats{
		Topic: topic,
	}

	// Get questions for this topic
	questions := getQuestionsForTopic(topic)
	stats.TotalQuestions = len(questions)

	// Track question text for duplicate detection
	seenQuestions := make(map[string][]int)

	for _, q := range questions {
		// 1. Validate CorrectAnswer index
		if q.CorrectAnswer < 0 || q.CorrectAnswer > 3 {
			errors = append(errors, ValidationError{
				Topic:      topic,
				QuestionID: q.ID,
				ErrorType:  "INVALID_INDEX",
				Message:    fmt.Sprintf("CorrectAnswer index %d is out of range (must be 0-3)", q.CorrectAnswer),
			})
		}

		// 2. Check for empty options
		for i, opt := range q.Options {
			if strings.TrimSpace(opt) == "" {
				errors = append(errors, ValidationError{
					Topic:      topic,
					QuestionID: q.ID,
					ErrorType:  "EMPTY_OPTION",
					Message:    fmt.Sprintf("Option %d is empty", i),
				})
			}
		}

		// 3. Check explanation shape
		expl := strings.TrimSpace(q.Explanation)
		if expl == "" {
			errors = append(errors, ValidationError{
				Topic:      topic,
				QuestionID: q.ID,
				ErrorType:  "EMPTY_EXPLANATION",
				Message:    "Explanation is empty",
			})
		} else {
			// 3a. Minimum length (40 chars) - flags fragmentary explanations
			if len([]rune(expl)) < 40 {
				errors = append(errors, ValidationError{
					Topic:      topic,
					QuestionID: q.ID,
					ErrorType:  "SHORT_EXPLANATION",
					Message:    fmt.Sprintf("Explanation is %d chars (< 40): %q", len([]rune(expl)), expl),
				})
			}
			// 3b. Must end in sentence punctuation
			last := expl[len(expl)-1]
			if last != '.' && last != '?' && last != '!' && last != ')' && last != '"' {
				errors = append(errors, ValidationError{
					Topic:      topic,
					QuestionID: q.ID,
					ErrorType:  "UNPUNCTUATED_EXPLANATION",
					Message:    fmt.Sprintf("Explanation does not end in sentence punctuation: %q", expl),
				})
			}
			// 3c. Broken decimal / abbreviation pattern introduced by an older
			// mechanical "Step N" splitter: e.g. "(127. Step 2: 0." or
			// "e. Step 3: g." — a single alphanumeric followed by ". Step N:".
			if brokenSplitRe.MatchString(expl) {
				errors = append(errors, ValidationError{
					Topic:      topic,
					QuestionID: q.ID,
					ErrorType:  "BROKEN_DECIMAL_SPLIT",
					Message:    fmt.Sprintf("Explanation looks like a broken decimal/abbreviation split: %q", expl),
				})
			}
			// 3d. "Step 1/2/3:" scaffolding is being removed in favour of
			// flowing prose; flag any remaining occurrences so the rewrite
			// backlog is visible.
			if stepScaffoldRe.MatchString(expl) {
				errors = append(errors, ValidationError{
					Topic:      topic,
					QuestionID: q.ID,
					ErrorType:  "STEP_SCAFFOLD",
					Message:    "Explanation still uses 'Step N:' scaffolding",
				})
			}
		}

		// 4. Track duplicates
		normalizedText := strings.ToLower(strings.TrimSpace(q.Text))
		seenQuestions[normalizedText] = append(seenQuestions[normalizedText], q.ID)

		// 5. Count "All of the above" usage
		for _, opt := range q.Options {
			optLower := strings.ToLower(strings.TrimSpace(opt))
			if strings.Contains(optLower, "all of the above") ||
				strings.Contains(optLower, "all the above") ||
				strings.Contains(optLower, "all above") {
				stats.AllOfAboveCount++
				break
			}
		}
	}

	// 6. Report duplicates
	for _, ids := range seenQuestions {
		if len(ids) > 1 {
			stats.DuplicateCount += len(ids) - 1
			errors = append(errors, ValidationError{
				Topic:      topic,
				QuestionID: ids[0],
				ErrorType:  "DUPLICATE",
				Message:    fmt.Sprintf("Question text appears %d times (IDs: %v)", len(ids), ids),
			})
		}
	}

	// Calculate percentages
	if stats.TotalQuestions > 0 {
		stats.AllOfAbovePercent = float64(stats.AllOfAboveCount) / float64(stats.TotalQuestions) * 100
	}

	// 7. Flag excessive "All of the above" usage (>20%)
	if stats.AllOfAbovePercent > 20.0 {
		errors = append(errors, ValidationError{
			Topic:      topic,
			QuestionID: 0,
			ErrorType:  "EXCESSIVE_ALL_OF_ABOVE",
			Message:    fmt.Sprintf("%.1f%% of questions use 'All of the above' (threshold: 20%%)", stats.AllOfAbovePercent),
		})
	}

	stats.Errors = len(errors)
	return errors, stats
}

func getQuestionsForTopic(topic string) []problems.Question {
	switch topic {
	case "algorithms":
		return problems.GetAlgorithmsQuestions()
	case "aws":
		return problems.GetAWSQuestions()
	case "azure":
		return problems.GetAzureQuestions()
	case "computer_architecture":
		return problems.GetComputerArchitectureQuestions()
	case "devops":
		return problems.GetDevOpsQuestions()
	case "frontend":
		return problems.GetFrontendQuestions()
	case "golang":
		return problems.GetGolangQuestions()
	case "kubernetes":
		return problems.GetKubernetesQuestions()
	case "linux":
		return problems.GetLinuxQuestions()
	case "machine_learning":
		return problems.GetMachineLearningQuestions()
	case "math":
		return problems.GetMathQuestions()
	case "networking":
		return problems.GetNetworkingQuestions()
	case "python":
		return problems.GetPythonQuestions()
	case "software_architecture":
		return problems.GetSoftwareArchitectureQuestions()
	case "systems_design":
		return problems.GetSystemsDesignQuestions()
	default:
		return []problems.Question{}
	}
}
