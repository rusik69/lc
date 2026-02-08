package problems

// Registration functions for course modules

// RegisterAlgorithmsModules registers algorithms course modules
func RegisterAlgorithmsModules(modules []CourseModule) {
	modulesMu.Lock()
	defer modulesMu.Unlock()
	allModules = append(allModules, modules...)
}

// RegisterSystemsDesignModules registers systems design course modules
func RegisterSystemsDesignModules(modules []CourseModule) {
	systemsDesignMu.Lock()
	defer systemsDesignMu.Unlock()
	allSystemsDesignModules = append(allSystemsDesignModules, modules...)
}

// RegisterGolangModules registers Golang course modules
func RegisterGolangModules(modules []CourseModule) {
	golangMu.Lock()
	defer golangMu.Unlock()
	allGolangModules = append(allGolangModules, modules...)
}

// RegisterPythonModules registers Python course modules
func RegisterPythonModules(modules []CourseModule) {
	pythonMu.Lock()
	defer pythonMu.Unlock()
	allPythonModules = append(allPythonModules, modules...)
}

// RegisterKubernetesModules registers Kubernetes course modules
func RegisterKubernetesModules(modules []CourseModule) {
	kubernetesMu.Lock()
	defer kubernetesMu.Unlock()
	allKubernetesModules = append(allKubernetesModules, modules...)
}

// RegisterMachineLearningModules registers machine learning course modules
func RegisterMachineLearningModules(modules []CourseModule) {
	machineLearningMu.Lock()
	defer machineLearningMu.Unlock()
	allMachineLearningModules = append(allMachineLearningModules, modules...)
}

// RegisterLinuxModules registers Linux course modules
func RegisterLinuxModules(modules []CourseModule) {
	linuxMu.Lock()
	defer linuxMu.Unlock()
	allLinuxModules = append(allLinuxModules, modules...)
}

// RegisterNetworkingModules registers networking course modules
func RegisterNetworkingModules(modules []CourseModule) {
	networkingMu.Lock()
	defer networkingMu.Unlock()
	allNetworkingModules = append(allNetworkingModules, modules...)
}

// RegisterFrontendModules registers frontend course modules
func RegisterFrontendModules(modules []CourseModule) {
	frontendMu.Lock()
	defer frontendMu.Unlock()
	allFrontendModules = append(allFrontendModules, modules...)
}

// RegisterDevOpsModules registers DevOps course modules
func RegisterDevOpsModules(modules []CourseModule) {
	devopsMu.Lock()
	defer devopsMu.Unlock()
	allDevOpsModules = append(allDevOpsModules, modules...)
}

// RegisterSoftwareArchitectureModules registers software architecture course modules
func RegisterSoftwareArchitectureModules(modules []CourseModule) {
	softwareArchitectureMu.Lock()
	defer softwareArchitectureMu.Unlock()
	allSoftwareArchitectureModules = append(allSoftwareArchitectureModules, modules...)
}

// RegisterAWSModules registers AWS course modules
func RegisterAWSModules(modules []CourseModule) {
	awsMu.Lock()
	defer awsMu.Unlock()
	allAWSModules = append(allAWSModules, modules...)
}

// RegisterComputerArchitectureModules registers computer architecture course modules
func RegisterComputerArchitectureModules(modules []CourseModule) {
	computerArchitectureMu.Lock()
	defer computerArchitectureMu.Unlock()
	allComputerArchitectureModules = append(allComputerArchitectureModules, modules...)
}

// RegisterAzureModules registers Azure course modules
func RegisterAzureModules(modules []CourseModule) {
	azureMu.Lock()
	defer azureMu.Unlock()
	allAzureModules = append(allAzureModules, modules...)
}

// RegisterMathModules registers math course modules
func RegisterMathModules(modules []CourseModule) {
	mathMu.Lock()
	defer mathMu.Unlock()
	allMathModules = append(allMathModules, modules...)
}

// Registration functions for course test questions

// RegisterGolangQuestions registers Golang test questions
func RegisterGolangQuestions(questions []Question) {
	golangQuestionsMu.Lock()
	defer golangQuestionsMu.Unlock()
	allGolangQuestions = append(allGolangQuestions, questions...)
}

// RegisterPythonQuestions registers Python test questions
func RegisterPythonQuestions(questions []Question) {
	pythonQuestionsMu.Lock()
	defer pythonQuestionsMu.Unlock()
	allPythonQuestions = append(allPythonQuestions, questions...)
}

// RegisterAlgorithmsQuestions registers algorithms test questions
func RegisterAlgorithmsQuestions(questions []Question) {
	algorithmsQuestionsMu.Lock()
	defer algorithmsQuestionsMu.Unlock()
	allAlgorithmsQuestions = append(allAlgorithmsQuestions, questions...)
}

// RegisterKubernetesQuestions registers Kubernetes test questions
func RegisterKubernetesQuestions(questions []Question) {
	kubernetesQuestionsMu.Lock()
	defer kubernetesQuestionsMu.Unlock()
	allKubernetesQuestions = append(allKubernetesQuestions, questions...)
}

// RegisterLinuxQuestions registers Linux test questions
func RegisterLinuxQuestions(questions []Question) {
	linuxQuestionsMu.Lock()
	defer linuxQuestionsMu.Unlock()
	allLinuxQuestions = append(allLinuxQuestions, questions...)
}

// RegisterAWSQuestions registers AWS test questions
func RegisterAWSQuestions(questions []Question) {
	awsQuestionsMu.Lock()
	defer awsQuestionsMu.Unlock()
	allAWSQuestions = append(allAWSQuestions, questions...)
}

// RegisterAzureQuestions registers Azure test questions
func RegisterAzureQuestions(questions []Question) {
	azureQuestionsMu.Lock()
	defer azureQuestionsMu.Unlock()
	allAzureQuestions = append(allAzureQuestions, questions...)
}

// RegisterDevOpsQuestions registers DevOps test questions
func RegisterDevOpsQuestions(questions []Question) {
	devopsQuestionsMu.Lock()
	defer devopsQuestionsMu.Unlock()
	allDevOpsQuestions = append(allDevOpsQuestions, questions...)
}

// RegisterFrontendQuestions registers frontend test questions
func RegisterFrontendQuestions(questions []Question) {
	frontendQuestionsMu.Lock()
	defer frontendQuestionsMu.Unlock()
	allFrontendQuestions = append(allFrontendQuestions, questions...)
}

// RegisterNetworkingQuestions registers networking test questions
func RegisterNetworkingQuestions(questions []Question) {
	networkingQuestionsMu.Lock()
	defer networkingQuestionsMu.Unlock()
	allNetworkingQuestions = append(allNetworkingQuestions, questions...)
}

// RegisterSystemsDesignQuestions registers systems design test questions
func RegisterSystemsDesignQuestions(questions []Question) {
	systemsDesignQuestionsMu.Lock()
	defer systemsDesignQuestionsMu.Unlock()
	allSystemsDesignQuestions = append(allSystemsDesignQuestions, questions...)
}

// RegisterSoftwareArchitectureQuestions registers software architecture test questions
func RegisterSoftwareArchitectureQuestions(questions []Question) {
	softwareArchitectureQuestionsMu.Lock()
	defer softwareArchitectureQuestionsMu.Unlock()
	allSoftwareArchitectureQuestions = append(allSoftwareArchitectureQuestions, questions...)
}

// RegisterMachineLearningQuestions registers machine learning test questions
func RegisterMachineLearningQuestions(questions []Question) {
	machineLearningQuestionsMu.Lock()
	defer machineLearningQuestionsMu.Unlock()
	allMachineLearningQuestions = append(allMachineLearningQuestions, questions...)
}

// RegisterComputerArchitectureQuestions registers computer architecture test questions
func RegisterComputerArchitectureQuestions(questions []Question) {
	computerArchitectureQuestionsMu.Lock()
	defer computerArchitectureQuestionsMu.Unlock()
	allComputerArchitectureQuestions = append(allComputerArchitectureQuestions, questions...)
}

// RegisterMathQuestions registers math test questions
func RegisterMathQuestions(questions []Question) {
	mathQuestionsMu.Lock()
	defer mathQuestionsMu.Unlock()
	allMathQuestions = append(allMathQuestions, questions...)
}
