package tests

import (
	"testing"

	"github.com/rusik69/lc/internal/problems"
)

func TestGetProblem(t *testing.T) {
	tests := []struct {
		name     string
		id       int
		wantNil  bool
		wantName string
	}{
		{
			name:     "Valid problem ID 1",
			id:       1,
			wantNil:  false,
			wantName: "Two Sum",
		},
		{
			name:     "Valid problem ID 2",
			id:       2,
			wantNil:  false,
			wantName: "Reverse String",
		},
		{
			name:     "Valid problem ID 3",
			id:       3,
			wantNil:  false,
			wantName: "FizzBuzz",
		},
		{
			name:     "Valid problem ID 4",
			id:       4,
			wantNil:  false,
			wantName: "Palindrome Check",
		},
		{
			name:     "Valid problem ID 5",
			id:       5,
			wantNil:  false,
			wantName: "Fibonacci",
		},
		{
			name:    "Invalid problem ID 0",
			id:      0,
			wantNil: true,
		},
		{
			name:    "Invalid problem ID 61",
			id:      61,
			wantNil: true,
		},
		{
			name:    "Invalid problem ID -1",
			id:      -1,
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := problems.GetProblem(tt.id)
			if tt.wantNil {
				if got != nil {
					t.Errorf("problems.GetProblem(%d) = %v, want nil", tt.id, got)
				}
			} else {
				if got == nil {
					t.Errorf("problems.GetProblem(%d) = nil, want non-nil", tt.id)
					return
				}
				if got.Title != tt.wantName {
					t.Errorf("problems.GetProblem(%d).Title = %v, want %v", tt.id, got.Title, tt.wantName)
				}
				if got.ID != tt.id {
					t.Errorf("problems.GetProblem(%d).ID = %v, want %v", tt.id, got.ID, tt.id)
				}
			}
		})
	}
}

func TestGetAllProblems(t *testing.T) {
	problems := problems.GetAllProblems()

	if len(problems) != 60 {
		t.Errorf("problems.GetAllProblems() returned %d problems, want 60", len(problems))
	}

	// Verify all problems have required fields
	for i, p := range problems {
		if p.ID == 0 {
			t.Errorf("Problem %d has ID 0", i)
		}
		if p.Title == "" {
			t.Errorf("Problem %d has empty Title", i)
		}
		if p.Description == "" {
			t.Errorf("Problem %d has empty Description", i)
		}
		if len(p.TestCases) == 0 {
			t.Errorf("Problem %d has no TestCases", i)
		}
		if p.Solution == "" {
			t.Errorf("Problem %d has empty Solution", i)
		}
	}
}

func TestProblemStructure(t *testing.T) {
	problem := problems.GetProblem(1)
	if problem == nil {
		t.Fatal("problems.GetProblem(1) returned nil")
	}

	// Verify test cases have required fields
	for i, tc := range problem.TestCases {
		if tc.Input == "" {
			t.Errorf("TestCase %d has empty Input", i)
		}
		if tc.Expected == "" {
			t.Errorf("TestCase %d has empty Expected", i)
		}
	}
}
