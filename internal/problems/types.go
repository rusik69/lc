package problems

import (
	"strings"
	"sync"
)

// TestCase represents a single test case for a problem
type TestCase struct {
	Input    string
	Expected string
}

// Problem represents a coding problem with its test cases and solution
type Problem struct {
	ID            int
	Title         string
	Description   string
	Difficulty    string // "Easy", "Medium", "Hard"
	Topic         string // Data structure/algorithm topic (e.g., "Arrays", "Hash Tables")
	Signature     string
	TestCases     []TestCase
	Solution      string // Go solution
	PythonSolution string // Python solution
	Explanation   string
}

// Lesson represents educational content within a course module
type Lesson struct {
	Title        string
	Content      string // HTML/markdown content explaining the concept
	CodeExamples string // Optional code examples
}

// CourseModule represents a course module with lessons and related problems
type CourseModule struct {
	ID          int
	Title       string
	Description string
	Order       int // For sequencing modules
	Lessons     []Lesson
	ProblemIDs  []int // IDs of problems related to this module
}

var (
	allProblems            []Problem
	problemsMu             sync.RWMutex
	allModules             []CourseModule
	modulesMu              sync.RWMutex
	allSystemsDesignModules []CourseModule
	systemsDesignMu        sync.RWMutex
	allGolangModules       []CourseModule
	golangMu               sync.RWMutex
	allPythonModules       []CourseModule
	pythonMu               sync.RWMutex
	allKubernetesModules   []CourseModule
	kubernetesMu            sync.RWMutex
	allMachineLearningModules []CourseModule
	machineLearningMu        sync.RWMutex
	allLinuxModules          []CourseModule
	linuxMu                  sync.RWMutex
)

func init() {
	allProblems = make([]Problem, 0, 60)
	allProblems = append(allProblems, EasyProblems...)
	allProblems = append(allProblems, MediumProblems...)
	allProblems = append(allProblems, HardProblems...)
}

// GetProblem returns a problem by its ID
func GetProblem(id int) *Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	for i := range allProblems {
		if allProblems[i].ID == id {
			return &allProblems[i]
		}
	}
	return nil
}

// GetAllProblems returns all problems
func GetAllProblems() []Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	result := make([]Problem, len(allProblems))
	copy(result, allProblems)
	return result
}

// GetProblemsByDifficulty returns all problems filtered by difficulty
func GetProblemsByDifficulty(difficulty string) []Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	result := []Problem{}
	for i := range allProblems {
		if allProblems[i].Difficulty == difficulty {
			result = append(result, allProblems[i])
		}
	}
	return result
}

// GetEasyProblems returns all easy problems
func GetEasyProblems() []Problem {
	return GetProblemsByDifficulty("Easy")
}

// GetMediumProblems returns all medium problems
func GetMediumProblems() []Problem {
	return GetProblemsByDifficulty("Medium")
}

// GetHardProblems returns all hard problems
func GetHardProblems() []Problem {
	return GetProblemsByDifficulty("Hard")
}

// GetSolution returns the solution for a problem in the specified language
func (p *Problem) GetSolution(language string) string {
	if language == "python" && p.PythonSolution != "" {
		return p.PythonSolution
	}
	return p.Solution
}

// HasLanguageSolution checks if a problem has a solution in the specified language
func (p *Problem) HasLanguageSolution(language string) bool {
	if language == "python" {
		return p.PythonSolution != ""
	}
	return p.Solution != ""
}

// GetSignature returns the signature for a problem in the specified language
func (p *Problem) GetSignature(language string) string {
	if language == "python" {
		return p.goToPythonSignature(p.Signature)
	}
	return p.Signature
}

// goToPythonSignature converts a Go function signature to Python signature
func (p *Problem) goToPythonSignature(goSig string) string {
	// Remove 'func ' prefix
	sig := goSig
	if len(sig) >= 5 && sig[:5] == "func " {
		sig = sig[5:]
	}
	
	// Extract function name (everything before first '(')
	funcNameEnd := 0
	for i, r := range sig {
		if r == '(' {
			funcNameEnd = i
			break
		}
	}
	if funcNameEnd == 0 {
		return "def solution():\n    pass"
	}
	
	funcName := strings.TrimSpace(sig[:funcNameEnd])
	
	// Extract parameters
	paramsStart := funcNameEnd
	paramsEnd := strings.Index(sig[paramsStart:], ")")
	if paramsEnd == -1 {
		return "def " + funcName + "():\n    pass"
	}
	paramsEnd += paramsStart
	
	paramsStr := strings.TrimSpace(sig[paramsStart+1 : paramsEnd])
	var params []string
	
	if paramsStr != "" {
		paramParts := strings.Split(paramsStr, ",")
		for _, part := range paramParts {
			part = strings.TrimSpace(part)
			fields := strings.Fields(part)
			if len(fields) >= 2 {
				paramName := fields[0]
				paramType := fields[1]
				
				// Convert Go types to Python types (order matters - do complex types first)
				pyType := paramType
				pyType = strings.ReplaceAll(pyType, "[]*ListNode", "List[Optional[ListNode]]")
				pyType = strings.ReplaceAll(pyType, "[]*TreeNode", "List[Optional[TreeNode]]")
				pyType = strings.ReplaceAll(pyType, "[]int", "List[int]")
				pyType = strings.ReplaceAll(pyType, "[]string", "List[str]")
				pyType = strings.ReplaceAll(pyType, "[]byte", "List[str]")
				pyType = strings.ReplaceAll(pyType, "*ListNode", "Optional[ListNode]")
				pyType = strings.ReplaceAll(pyType, "*TreeNode", "Optional[TreeNode]")
				pyType = strings.ReplaceAll(pyType, "float64", "float")
				pyType = strings.ReplaceAll(pyType, "string", "str")
				pyType = strings.ReplaceAll(pyType, "bool", "bool")
				// Keep int as int (no change needed)
				
				params = append(params, paramName+": "+pyType)
			}
		}
	}
	
	// Extract return type
	returnType := "None"
	remaining := strings.TrimSpace(sig[paramsEnd+1:])
	if strings.Contains(remaining, "[]int") {
		returnType = "List[int]"
	} else if strings.Contains(remaining, "[]string") {
		returnType = "List[str]"
	} else if strings.Contains(remaining, "string") {
		returnType = "str"
	} else if strings.Contains(remaining, "int") {
		returnType = "int"
	} else if strings.Contains(remaining, "bool") {
		returnType = "bool"
	} else if strings.Contains(remaining, "float64") {
		returnType = "float"
	} else if strings.Contains(remaining, "*ListNode") {
		returnType = "Optional[ListNode]"
	} else if strings.Contains(remaining, "*TreeNode") {
		returnType = "Optional[TreeNode]"
	}
	
	return "def " + funcName + "(" + strings.Join(params, ", ") + ") -> " + returnType
}

// GetProblemsByTopic returns all problems filtered by topic
func GetProblemsByTopic(topic string) []Problem {
	problemsMu.RLock()
	defer problemsMu.RUnlock()
	result := []Problem{}
	for i := range allProblems {
		if allProblems[i].Topic == topic {
			result = append(result, allProblems[i])
		}
	}
	return result
}

// GetCourseModules returns all course modules
func GetCourseModules() []CourseModule {
	modulesMu.RLock()
	defer modulesMu.RUnlock()
	result := make([]CourseModule, len(allModules))
	copy(result, allModules)
	return result
}

// GetModuleByID returns a course module by its ID
func GetModuleByID(id int) *CourseModule {
	modulesMu.RLock()
	defer modulesMu.RUnlock()
	for i := range allModules {
		if allModules[i].ID == id {
			return &allModules[i]
		}
	}
	return nil
}

// GetSystemsDesignModules returns all systems design course modules
func GetSystemsDesignModules() []CourseModule {
	systemsDesignMu.RLock()
	defer systemsDesignMu.RUnlock()
	result := make([]CourseModule, len(allSystemsDesignModules))
	copy(result, allSystemsDesignModules)
	return result
}

// GetSystemsDesignModuleByID returns a systems design module by its ID
func GetSystemsDesignModuleByID(id int) *CourseModule {
	systemsDesignMu.RLock()
	defer systemsDesignMu.RUnlock()
	for i := range allSystemsDesignModules {
		if allSystemsDesignModules[i].ID == id {
			return &allSystemsDesignModules[i]
		}
	}
	return nil
}

// GetGolangModules returns all Golang course modules
func GetGolangModules() []CourseModule {
	golangMu.RLock()
	defer golangMu.RUnlock()
	result := make([]CourseModule, len(allGolangModules))
	copy(result, allGolangModules)
	return result
}

// GetGolangModuleByID returns a Golang module by its ID
func GetGolangModuleByID(id int) *CourseModule {
	golangMu.RLock()
	defer golangMu.RUnlock()
	for i := range allGolangModules {
		if allGolangModules[i].ID == id {
			return &allGolangModules[i]
		}
	}
	return nil
}

// GetPythonModules returns all Python course modules
func GetPythonModules() []CourseModule {
	pythonMu.RLock()
	defer pythonMu.RUnlock()
	result := make([]CourseModule, len(allPythonModules))
	copy(result, allPythonModules)
	return result
}

// GetPythonModuleByID returns a Python module by its ID
func GetPythonModuleByID(id int) *CourseModule {
	pythonMu.RLock()
	defer pythonMu.RUnlock()
	for i := range allPythonModules {
		if allPythonModules[i].ID == id {
			return &allPythonModules[i]
		}
	}
	return nil
}

// GetKubernetesModules returns all Kubernetes course modules
func GetKubernetesModules() []CourseModule {
	kubernetesMu.RLock()
	defer kubernetesMu.RUnlock()
	result := make([]CourseModule, len(allKubernetesModules))
	copy(result, allKubernetesModules)
	return result
}

// GetKubernetesModuleByID returns a Kubernetes module by its ID
func GetKubernetesModuleByID(id int) *CourseModule {
	kubernetesMu.RLock()
	defer kubernetesMu.RUnlock()
	for i := range allKubernetesModules {
		if allKubernetesModules[i].ID == id {
			return &allKubernetesModules[i]
		}
	}
	return nil
}

// GetMachineLearningModules returns all machine learning course modules
func GetMachineLearningModules() []CourseModule {
	machineLearningMu.RLock()
	defer machineLearningMu.RUnlock()
	result := make([]CourseModule, len(allMachineLearningModules))
	copy(result, allMachineLearningModules)
	return result
}

// GetMachineLearningModuleByID returns a machine learning module by its ID
func GetMachineLearningModuleByID(id int) *CourseModule {
	machineLearningMu.RLock()
	defer machineLearningMu.RUnlock()
	for i := range allMachineLearningModules {
		if allMachineLearningModules[i].ID == id {
			return &allMachineLearningModules[i]
		}
	}
	return nil
}

// GetLinuxModules returns all Linux course modules
func GetLinuxModules() []CourseModule {
	linuxMu.RLock()
	defer linuxMu.RUnlock()
	result := make([]CourseModule, len(allLinuxModules))
	copy(result, allLinuxModules)
	return result
}

// GetLinuxModuleByID returns a Linux module by its ID
func GetLinuxModuleByID(id int) *CourseModule {
	linuxMu.RLock()
	defer linuxMu.RUnlock()
	for i := range allLinuxModules {
		if allLinuxModules[i].ID == id {
			return &allLinuxModules[i]
		}
	}
	return nil
}
