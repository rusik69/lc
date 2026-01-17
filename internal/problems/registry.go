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
