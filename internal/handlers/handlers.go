package handlers

import (
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gomarkdown/markdown"
	"github.com/gomarkdown/markdown/html"
	"github.com/gomarkdown/markdown/parser"
	"github.com/rusik69/lc/internal/executor"
	"github.com/rusik69/lc/internal/problems"
)

var (
	templates                *template.Template
	indexTemplate            *template.Template
	problemTemplate          *template.Template
	algorithmsTemplate       *template.Template
	algorithmsModuleTemplate *template.Template
	systemsDesignTemplate    *template.Template
	systemsDesignModuleTemplate *template.Template
	golangTemplate           *template.Template
	golangModuleTemplate     *template.Template
	pythonTemplate           *template.Template
	pythonModuleTemplate     *template.Template
	kubernetesTemplate       *template.Template
	kubernetesModuleTemplate *template.Template
	machineLearningTemplate  *template.Template
	machineLearningModuleTemplate *template.Template
	linuxTemplate            *template.Template
	linuxModuleTemplate      *template.Template
	networkingTemplate       *template.Template
	networkingModuleTemplate *template.Template
	frontendTemplate         *template.Template
	frontendModuleTemplate   *template.Template
	devopsTemplate           *template.Template
	devopsModuleTemplate     *template.Template
	softwareArchitectureTemplate *template.Template
	softwareArchitectureModuleTemplate *template.Template
	awsTemplate                  *template.Template
	awsModuleTemplate           *template.Template
	computerArchitectureTemplate *template.Template
	computerArchitectureModuleTemplate *template.Template
	azureTemplate               *template.Template
	azureModuleTemplate         *template.Template
)

const (
	maxCodeSize    = 100 * 1024 // 100KB max code size
	maxRequestSize = 150 * 1024 // 150KB max request size
)

// markdownToHTML converts markdown text to HTML
func markdownToHTML(md string) template.HTML {
	// Create markdown parser
	extensions := parser.CommonExtensions | parser.AutoHeadingIDs | parser.NoEmptyLineBeforeBlock
	p := parser.NewWithExtensions(extensions)
	
	// Parse markdown
	doc := p.Parse([]byte(md))
	
	// Create HTML renderer
	htmlFlags := html.CommonFlags | html.HrefTargetBlank
	opts := html.RendererOptions{Flags: htmlFlags}
	renderer := html.NewRenderer(opts)
	
	// Render to HTML
	htmlBytes := markdown.Render(doc, renderer)
	return template.HTML(htmlBytes)
}

// InitTemplates initializes HTML templates
func InitTemplates(templateDir string) error {
	// Create template function map
	funcMap := template.FuncMap{
		"markdown": markdownToHTML,
	}
	
	var err error
	templates, err = template.New("").Funcs(funcMap).ParseGlob(templateDir + "/*.html")
	if err != nil {
		return err
	}

	// Parse index template separately to avoid block conflicts
	indexTemplate, err = template.New("index").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/index.html")
	if err != nil {
		return err
	}

	// Parse problem template separately to avoid block conflicts
	problemTemplate, err = template.New("problem").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/problem.html")
	if err != nil {
		return err
	}

	// Parse algorithms template separately
	algorithmsTemplate, err = template.New("algorithms").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/algorithms.html")
	if err != nil {
		return err
	}

	// Parse algorithms module template separately
	algorithmsModuleTemplate, err = template.New("algorithms_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/algorithms_module.html")
	if err != nil {
		return err
	}

	// Parse systems design course template
	systemsDesignTemplate, err = template.New("systems_design").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/systems_design_course.html")
	if err != nil {
		return err
	}

	// Parse systems design module template
	systemsDesignModuleTemplate, err = template.New("systems_design_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/systems_design_module.html")
	if err != nil {
		return err
	}

	// Parse Golang course template
	golangTemplate, err = template.New("golang").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/golang_course.html")
	if err != nil {
		return err
	}

	// Parse Golang module template
	golangModuleTemplate, err = template.New("golang_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/golang_module.html")
	if err != nil {
		return err
	}

	// Parse Python course template
	pythonTemplate, err = template.New("python").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/python_course.html")
	if err != nil {
		return err
	}

	// Parse Python module template
	pythonModuleTemplate, err = template.New("python_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/python_module.html")
	if err != nil {
		return err
	}

	// Parse Kubernetes course template
	kubernetesTemplate, err = template.New("kubernetes").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/kubernetes_course.html")
	if err != nil {
		return err
	}

	// Parse Kubernetes module template
	kubernetesModuleTemplate, err = template.New("kubernetes_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/kubernetes_module.html")
	if err != nil {
		return err
	}

	// Parse Machine Learning course template
	machineLearningTemplate, err = template.New("machine_learning").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/machine_learning_course.html")
	if err != nil {
		return err
	}

	// Parse Machine Learning module template
	machineLearningModuleTemplate, err = template.New("machine_learning_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/machine_learning_module.html")
	if err != nil {
		return err
	}

	// Parse Linux course template
	linuxTemplate, err = template.New("linux").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/linux_course.html")
	if err != nil {
		return err
	}

	// Parse Linux module template
	linuxModuleTemplate, err = template.New("linux_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/linux_module.html")
	if err != nil {
		return err
	}

	// Parse Networking course template
	networkingTemplate, err = template.New("networking").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/networking_course.html")
	if err != nil {
		return err
	}

	// Parse Networking module template
	networkingModuleTemplate, err = template.New("networking_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/networking_module.html")
	if err != nil {
		return err
	}

	// Parse Frontend course template
	frontendTemplate, err = template.New("frontend").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/frontend_course.html")
	if err != nil {
		return err
	}

	// Parse Frontend module template
	frontendModuleTemplate, err = template.New("frontend_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/frontend_module.html")
	if err != nil {
		return err
	}

	// Parse DevOps course template
	devopsTemplate, err = template.New("devops").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/devops_course.html")
	if err != nil {
		return err
	}

	// Parse DevOps module template
	devopsModuleTemplate, err = template.New("devops_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/devops_module.html")
	if err != nil {
		return err
	}

	// Parse Software Architecture course template
	softwareArchitectureTemplate, err = template.New("software_architecture").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/software_architecture_course.html")
	if err != nil {
		return err
	}

	// Parse Software Architecture module template
	softwareArchitectureModuleTemplate, err = template.New("software_architecture_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/software_architecture_module.html")
	if err != nil {
		return err
	}

	// Parse AWS course template
	awsTemplate, err = template.New("aws").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/aws_course.html")
	if err != nil {
		return err
	}

	// Parse AWS module template
	awsModuleTemplate, err = template.New("aws_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/aws_module.html")
	if err != nil {
		return err
	}

	// Parse Computer Architecture course template
	computerArchitectureTemplate, err = template.New("computer_architecture").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/computer_architecture_course.html")
	if err != nil {
		return err
	}

	// Parse Computer Architecture module template
	computerArchitectureModuleTemplate, err = template.New("computer_architecture_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/computer_architecture_module.html")
	if err != nil {
		return err
	}

	// Parse Azure course template
	azureTemplate, err = template.New("azure").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/azure_course.html")
	if err != nil {
		return err
	}

	// Parse Azure module template
	azureModuleTemplate, err = template.New("azure_module").Funcs(funcMap).ParseFiles(templateDir+"/layout.html", templateDir+"/azure_module.html")
	if err != nil {
		return err
	}

	return nil
}

// rateLimiter provides simple rate limiting
type rateLimiter struct {
	requests map[string][]time.Time
	limit    int
	window   time.Duration
}

var limiter = &rateLimiter{
	requests: make(map[string][]time.Time),
	limit:    10,          // 10 requests
	window:   time.Minute, // per minute
}

func (rl *rateLimiter) allow(ip string) bool {
	now := time.Now()
	rl.requests[ip] = append(rl.requests[ip], now)

	// Remove old requests outside the window
	valid := []time.Time{}
	for _, t := range rl.requests[ip] {
		if now.Sub(t) < rl.window {
			valid = append(valid, t)
		}
	}
	rl.requests[ip] = valid

	return len(valid) <= rl.limit
}

func getClientIP(r *http.Request) string {
	ip := r.Header.Get("X-Forwarded-For")
	if ip != "" {
		parts := strings.Split(ip, ",")
		return strings.TrimSpace(parts[0])
	}
	ip = r.Header.Get("X-Real-Ip")
	if ip != "" {
		return ip
	}
	parts := strings.Split(r.RemoteAddr, ":")
	return parts[0]
}

func HandleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	problems := problems.GetAllProblems()
	if err := indexTemplate.ExecuteTemplate(w, "layout.html", problems); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleProblem(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	id, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	problem := problems.GetProblem(id)
	if problem == nil {
		http.NotFound(w, r)
		return
	}

	if err := problemTemplate.ExecuteTemplate(w, "layout.html", problem); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleRun(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	// Handle preflight requests
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Rate limiting
	if !limiter.allow(getClientIP(r)) {
		http.Error(w, "Rate limit exceeded", http.StatusTooManyRequests)
		return
	}

	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	id, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	problem := problems.GetProblem(id)
	if problem == nil {
		http.NotFound(w, r)
		return
	}

	// Check if streaming is requested (before parsing form)
	streamRequested := r.URL.Query().Get("stream") == "true"

	// Parse form with size limit
	// For multipart/form-data (FormData from fetch), use ParseMultipartForm
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(maxRequestSize); err != nil {
			errMsg := err.Error()
			if strings.Contains(errMsg, "request body too large") || strings.Contains(errMsg, "http: request body too large") {
				http.Error(w, "Request too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		// For application/x-www-form-urlencoded, limit body size first
		if !streamRequested {
			r.Body = http.MaxBytesReader(w, r.Body, maxRequestSize)
			defer r.Body.Close()
		}
		if err := r.ParseForm(); err != nil {
			errMsg := err.Error()
			if strings.Contains(errMsg, "request body too large") || strings.Contains(errMsg, "http: request body too large") {
				http.Error(w, "Request too large", http.StatusRequestEntityTooLarge)
				return
			}
			http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
			return
		}
	}

	userCode := r.FormValue("code")
	if userCode == "" {
		http.Error(w, "Code is required", http.StatusBadRequest)
		return
	}

	language := r.FormValue("language")
	if language == "" {
		language = "go" // Default to Go
	}

	// Validate code size
	if len(userCode) > maxCodeSize {
		http.Error(w, fmt.Sprintf("Code too large (max %d bytes)", maxCodeSize), http.StatusBadRequest)
		return
	}

	// Basic validation: ensure code doesn't contain dangerous patterns
	if containsDangerousPatterns(userCode) {
		http.Error(w, "Code contains prohibited patterns", http.StatusBadRequest)
		return
	}

	action := r.FormValue("action")
	runAllTests := action == "submit"

	if streamRequested && !runAllTests {
		// Stream output for Run button
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		// Send initial comment to establish connection
		fmt.Fprintf(w, ": connection established\n\n")
		flusher.Flush()

		// Ensure we always send the done event, even on panic
		defer func() {
			if r := recover(); r != nil {
				// Write error to stream before sending done event
				fmt.Fprintf(w, "data: %s\n\n", "[ERROR] Internal server error: "+fmt.Sprintf("%v", r))
				flusher.Flush()
				fmt.Fprintf(w, "event: done\ndata: {\"success\":false,\"passed\":0,\"total\":0}\n\n")
				flusher.Flush()
			}
		}()

		var result *executor.ExecutionResult
		func() {
			defer func() {
				if r := recover(); r != nil {
					// Write panic error to stream
					fmt.Fprintf(w, "data: %s\n\n", "[ERROR] Execution panic: "+fmt.Sprintf("%v", r))
					flusher.Flush()
					result = &executor.ExecutionResult{
						Success: false,
						Error:   fmt.Sprintf("Execution panic: %v", r),
						Results: []executor.TestResult{},
					}
				}
			}()
			result = executor.ExecuteCodeStreamWithLanguage(problem, userCode, runAllTests, w, language)
		}()

		// Ensure result is not nil
		if result == nil {
			result = &executor.ExecutionResult{
				Success: false,
				Error:   "Execution returned nil result",
				Results: []executor.TestResult{},
			}
		}

		passed := 0
		for _, r := range result.Results {
			if r.Passed {
				passed++
			}
		}

		// Always send done event, even if there were errors
		// Check if connection is still open before writing
		doneMsg := fmt.Sprintf("event: done\ndata: {\"success\":%t,\"passed\":%d,\"total\":%d}\n\n", result.Success, passed, len(result.Results))
		if _, err := fmt.Fprintf(w, doneMsg); err != nil {
			// Connection closed or timed out - log but don't fail
			// This is expected if client disconnects
			return
		}
		// Only flush if write succeeded
		flusher.Flush()
		return
	}

	// Non-streaming execution (for Submit button)
	result := executor.ExecuteCodeWithLanguage(problem, userCode, runAllTests, language)
	passed := 0
	for _, r := range result.Results {
		if r.Passed {
			passed++
		}
	}

	// If no results and no error, add a helpful message
	if len(result.Results) == 0 && result.Error == "" && result.Output == "" {
		result.Error = "No test results generated. The code may have compilation issues or the test execution failed."
		if result.Stderr != "" {
			result.Error += "\n\nStderr: " + result.Stderr
		}
	}

	data := struct {
		Problem *problems.Problem
		Result  *executor.ExecutionResult
		Passed  int
		Total   int
	}{
		Problem: problem,
		Result:  result,
		Passed:  passed,
		Total:   len(result.Results),
	}

	if err := templates.ExecuteTemplate(w, "result.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleSolution(w http.ResponseWriter, r *http.Request) {
	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	// Handle preflight requests
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	id, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	problem := problems.GetProblem(id)
	if problem == nil {
		http.NotFound(w, r)
		return
	}

	// Get language from query parameter, default to go
	language := r.URL.Query().Get("lang")
	if language == "" {
		language = "go"
	}

	solution := problem.GetSolution(language)
	codeLang := "go"
	if language == "python" && problem.HasLanguageSolution("python") {
		codeLang = "python"
	}

	explanationHTML := ""
	if problem.Explanation != "" {
		explanationHTML = `<div class="explanation"><h4>Explanation</h4><p>` + template.HTMLEscapeString(problem.Explanation) + `</p></div>`
	}
	
	// Add language selector
	goSelected := ""
	pythonSelected := ""
	if language == "go" {
		goSelected = " selected"
	} else {
		pythonSelected = " selected"
	}
	langSelector := `<div style="margin-bottom: 10px;">
		<label for="solution-lang" style="margin-right: 10px; color: #ccc;">Language:</label>
		<select id="solution-lang" onchange="loadSolution(` + fmt.Sprintf("%d", id) + `, this.value)" style="padding: 5px; background: #272822; color: #f8f8f2; border: 1px solid #555; border-radius: 3px;">
			<option value="go"` + goSelected + `>Go</option>
			<option value="python"` + pythonSelected + `>Python 3</option>
		</select>
	</div>`

	fmt.Fprintf(w, `<div class="solution">
	%s
	<h3>Solution</h3>
	<pre><code class="language-%s">%s</code></pre>
	%s
</div>
<script>
	if (typeof CodeMirror !== 'undefined') {
		setTimeout(function() {
			var solutionCode = document.querySelector('.solution code');
			if (solutionCode) {
				CodeMirror.runMode(solutionCode.textContent, '%s', solutionCode);
			}
		}, 100);
	}
</script>`, langSelector, codeLang, template.HTMLEscapeString(solution), explanationHTML, codeLang)
}

func HandleAlgorithms(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/algorithms" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetCourseModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := algorithmsTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleAlgorithmsModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := algorithmsModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleSystemsDesignCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/systems-design" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetSystemsDesignModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := systemsDesignTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleSystemsDesignModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetSystemsDesignModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module (empty for now)
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := systemsDesignModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleGolangCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/golang" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetGolangModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := golangTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleGolangModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetGolangModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := golangModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandlePythonCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/python" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetPythonModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := pythonTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandlePythonModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetPythonModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := pythonModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleKubernetesCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/kubernetes" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetKubernetesModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := kubernetesTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleKubernetesModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetKubernetesModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := kubernetesModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleMachineLearningCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/machine-learning" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetMachineLearningModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := machineLearningTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleMachineLearningModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetMachineLearningModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := machineLearningModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleLinuxCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/linux" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetLinuxModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := linuxTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleLinuxModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetLinuxModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := linuxModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleNetworkingCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/networking" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetNetworkingModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := networkingTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleNetworkingModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetNetworkingModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := networkingModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleFrontendCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/frontend" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetFrontendModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := frontendTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleFrontendModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetFrontendModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := frontendModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleDevOpsCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/devops" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetDevOpsModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := devopsTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleDevOpsModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetDevOpsModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := devopsModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleSoftwareArchitectureCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/software-architecture" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetSoftwareArchitectureModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := softwareArchitectureTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleSoftwareArchitectureModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetSoftwareArchitectureModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := softwareArchitectureModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleAWSCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/aws" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetAWSModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := awsTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleAWSModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetAWSModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := awsModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleAzureCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/azure" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetAzureModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := azureTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleAzureModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetAzureModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := azureModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleComputerArchitectureCourse(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/computer-architecture" {
		http.NotFound(w, r)
		return
	}

	modules := problems.GetComputerArchitectureModules()
	// Deduplicate modules by ID
	seen := make(map[int]bool)
	deduplicated := make([]problems.CourseModule, 0, len(modules))
	for _, module := range modules {
		if !seen[module.ID] {
			seen[module.ID] = true
			deduplicated = append(deduplicated, module)
		}
	}
	// Sort modules by Order to ensure correct display order
	sort.Slice(deduplicated, func(i, j int) bool {
		return deduplicated[i].Order < deduplicated[j].Order
	})
	if err := computerArchitectureTemplate.ExecuteTemplate(w, "layout.html", deduplicated); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func HandleComputerArchitectureModule(w http.ResponseWriter, r *http.Request) {
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 3 {
		http.NotFound(w, r)
		return
	}

	moduleID, err := strconv.Atoi(parts[2])
	if err != nil {
		http.NotFound(w, r)
		return
	}

	module := problems.GetComputerArchitectureModuleByID(moduleID)
	if module == nil {
		http.NotFound(w, r)
		return
	}

	// Get problems for this module
	var moduleProblems []problems.Problem
	for _, problemID := range module.ProblemIDs {
		problem := problems.GetProblem(problemID)
		if problem != nil {
			moduleProblems = append(moduleProblems, *problem)
		}
	}

	data := struct {
		Module   *problems.CourseModule
		Problems []problems.Problem
	}{
		Module:   module,
		Problems: moduleProblems,
	}

	if err := computerArchitectureModuleTemplate.ExecuteTemplate(w, "layout.html", data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func containsDangerousPatterns(code string) bool {
	dangerous := []string{
		"os.Exec",
		"os/exec",
		"syscall",
		"unsafe",
		"runtime",
		"import \"C\"",
		"cgo",
		"net/http",
		"net/url",
		"io/ioutil",
		"os.Open",
		"os.Create",
		"os.Remove",
		"os.Mkdir",
		"os.Chmod",
		"os.Chown",
		"exec.Command",
		"exec.Run",
	}

	codeLower := strings.ToLower(code)
	for _, pattern := range dangerous {
		if strings.Contains(codeLower, strings.ToLower(pattern)) {
			return true
		}
	}
	return false
}
