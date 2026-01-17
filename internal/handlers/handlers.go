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

		result := executor.ExecuteCodeStreamWithLanguage(problem, userCode, runAllTests, w, language)

		passed := 0
		for _, r := range result.Results {
			if r.Passed {
				passed++
			}
		}

		fmt.Fprintf(w, "event: done\ndata: {\"success\":%t,\"passed\":%d,\"total\":%d}\n\n", result.Success, passed, len(result.Results))
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
