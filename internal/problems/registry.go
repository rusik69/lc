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
