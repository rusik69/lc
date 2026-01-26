package main

import (
	"log"
	"net/http"
	"time"

	"github.com/rusik69/lc/internal/auth"
	_ "github.com/rusik69/lc/internal/courses/algorithms"
	_ "github.com/rusik69/lc/internal/courses/golang"
	_ "github.com/rusik69/lc/internal/courses/kubernetes"
	_ "github.com/rusik69/lc/internal/courses/linux"
	_ "github.com/rusik69/lc/internal/courses/machine_learning"
	_ "github.com/rusik69/lc/internal/courses/networking"
	_ "github.com/rusik69/lc/internal/courses/python"
	_ "github.com/rusik69/lc/internal/courses/systems_design"
	_ "github.com/rusik69/lc/internal/courses/frontend"
	_ "github.com/rusik69/lc/internal/courses/devops"
	_ "github.com/rusik69/lc/internal/courses/software_architecture"
	_ "github.com/rusik69/lc/internal/courses/aws"
	_ "github.com/rusik69/lc/internal/courses/computer_architecture"
	_ "github.com/rusik69/lc/internal/courses/azure"
	_ "github.com/rusik69/lc/internal/courses/math"
	"github.com/rusik69/lc/internal/handlers"
)

func main() {
	// Initialize auth
	if err := auth.InitAuth(); err != nil {
		log.Fatalf("Failed to initialize auth: %v", err)
	}

	// Initialize templates
	if err := handlers.InitTemplates("web/templates"); err != nil {
		log.Fatalf("Failed to load templates: %v", err)
	}

	mux := http.NewServeMux()

	// Auth routes (no auth required)
	mux.HandleFunc("/login", auth.HandleLogin)
	mux.HandleFunc("/logout", auth.HandleLogout)

	// Protected routes - exact paths first, then paths with trailing slashes
	mux.HandleFunc("/", handlers.HandleIndex)
	mux.HandleFunc("/algorithms", handlers.HandleAlgorithms)
	mux.HandleFunc("/algorithms/", handlers.HandleAlgorithmsModule)
	mux.HandleFunc("/systems-design", handlers.HandleSystemsDesignCourse)
	mux.HandleFunc("/systems-design/", handlers.HandleSystemsDesignModule)
	mux.HandleFunc("/golang", handlers.HandleGolangCourse)
	mux.HandleFunc("/golang/", handlers.HandleGolangModule)
	mux.HandleFunc("/python", handlers.HandlePythonCourse)
	mux.HandleFunc("/python/", handlers.HandlePythonModule)
	mux.HandleFunc("/kubernetes", handlers.HandleKubernetesCourse)
	mux.HandleFunc("/kubernetes/", handlers.HandleKubernetesModule)
	mux.HandleFunc("/linux", handlers.HandleLinuxCourse)
	mux.HandleFunc("/linux/", handlers.HandleLinuxModule)
	mux.HandleFunc("/machine-learning", handlers.HandleMachineLearningCourse)
	mux.HandleFunc("/machine-learning/", handlers.HandleMachineLearningModule)
	mux.HandleFunc("/networking", handlers.HandleNetworkingCourse)
	mux.HandleFunc("/networking/", handlers.HandleNetworkingModule)
	mux.HandleFunc("/frontend", handlers.HandleFrontendCourse)
	mux.HandleFunc("/frontend/", handlers.HandleFrontendModule)
	mux.HandleFunc("/devops", handlers.HandleDevOpsCourse)
	mux.HandleFunc("/devops/", handlers.HandleDevOpsModule)
	mux.HandleFunc("/software-architecture", handlers.HandleSoftwareArchitectureCourse)
	mux.HandleFunc("/software-architecture/", handlers.HandleSoftwareArchitectureModule)
	mux.HandleFunc("/aws", handlers.HandleAWSCourse)
	mux.HandleFunc("/aws/", handlers.HandleAWSModule)
	mux.HandleFunc("/azure", handlers.HandleAzureCourse)
	mux.HandleFunc("/azure/", handlers.HandleAzureModule)
	mux.HandleFunc("/computer-architecture", handlers.HandleComputerArchitectureCourse)
	mux.HandleFunc("/computer-architecture/", handlers.HandleComputerArchitectureModule)
	mux.HandleFunc("/math", handlers.HandleMathCourse)
	mux.HandleFunc("/math/", handlers.HandleMathModule)
	mux.HandleFunc("/problem/", handlers.HandleProblem)
	mux.HandleFunc("/run/", handlers.HandleRun)
	mux.HandleFunc("/solution/", handlers.HandleSolution)
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("web/static"))))

	// Apply auth middleware
	handler := auth.Middleware(mux)

	server := &http.Server{
		Addr:           ":8080",
		Handler:        handler,
		ReadTimeout:    15 * time.Second,
		WriteTimeout:   60 * time.Second, // Increased for SSE streaming (code execution can take up to 25s + overhead)
		IdleTimeout:    120 * time.Second, // Increased for long-lived SSE connections
		MaxHeaderBytes: 1 << 20, // 1MB
	}

	log.Println("Server starting on :8080")
	log.Fatal(server.ListenAndServe())
}
