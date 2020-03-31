#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>     // Basic input output
#include <stdexcept>    // Errors
#include <functional>   // Lambda Functions -> Resource management
#include <vector>       // Container
#include <cstdlib>      // Macros -> EXIT_SUCCESS 
#include <cstring>      // Strings
#include <optional>     // Alows for seperating empty / 0 on a return, std::optionl is empty till assigned
#include <map>          // No explored yet

const int WIDTH = 800;      // Screen deetz
const int HEIGHT = 600;

const std::vector<const char*> validationLayers = {         // Needed (coming back to Valid layers, fix then)
	"VK_LAYER_KHRONOS_validation"
};

// This removes extra checks and work when we are in release and we have removed all bugs(as if)
#ifdef NDEBUG                               // Part of STL, if Debug, else
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class HelloTriangleApplication {            // Seperated into a better structure later
public:
	void run() {
		initWindow();                       // Setup GLFW window and settings for Vulkan
		initVulkan();                       // Setting up the information and checks needed for the drivers
		mainLoop();                         // Loop
		cleanup();                          // Clearing memory off of the heap
	}

private:

	GLFWwindow* window;                     // Main window

	VkInstance instance;                    // Instance of Vulkan, needed for everything.

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;       // Hold information about the GPU device, set to null

	// There are many queue families
	// From which speific queues are derived to manage code
	// This struct is used to check if they are available to the GPU
	// Also to hold information about the Q's
	struct QueueFamilyIndices {
		std::optional<uint32_t> graphicsFamily;          // Optional is empty till assigned (So it won't mix up index 0 from nothing)

		bool isComplete() {
			return graphicsFamily.has_value();           // Bool
		}
	};

	void initVulkan() {
		createInstance();        // Needed first

		pickPhysicalDevice();    // Pick a GPU, you can use many (I am not) this picks which one based on checks  
	}
	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport()) 
		{                         // If it's support and in Debug
			throw std::runtime_error("validation layers requested, but not available!");
		}

		// To allow Vulkan to be generic a lot of the functionalisty is abstracted away into extentions
		// Different GPUs can support some of these
		uint32_t extensionCount = 0;                                                          // Create some memory to count them                                   
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);            // Layer, memory for count, memory for assignment (I use default layer for now)
		std::vector<VkExtensionProperties> extensions(extensionCount);                        // Make a vector with the exact size of the count
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());  // Call again with container

		// Outputting then to console
		std::cout << "available extensions:" << std::endl;
		for (const auto& extension : extensions) {
			std::cout << "\t" << extension.extensionName << std::endl; // \t = tab
		}

		// These next 2 go hand in hand, app info first
		VkApplicationInfo appInfo = {}; // Defalt init, leaves .pnext as nullptr, pnext tells the memory where to go next after use, nullptr deletes it
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;  // Most structs have this, like an ID
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo; // Apllies info from above

		// Come back to the layers
		if (enableValidationLayers) 
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else 
		{
			createInfo.enabledLayerCount = 0;
		}

		uint32_t glfwExtensionCount = 0;    // How many, passed as memeory and counted
		const char** glfwExtensions;        // Name list is returned

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);    // Returns extension list and creates count

		createInfo.enabledExtensionCount = glfwExtensionCount;      // This info is added to the info struct
		createInfo.ppEnabledExtensionNames = glfwExtensions;

		// Global Validation Layer
		// Global means on whole program not a specific device
		createInfo.enabledLayerCount = 0;  // Empty for now

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) // Creates instance
		{
			throw std::runtime_error("failed to create instance!");
		}
	}


	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);    // Pass in memory for Number of GPU Devices
		if (deviceCount == 0) 
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");   // If none we leave
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());     // Passes in container to hold all the devices

		for (const auto& device : devices) 
		{        // Of our GPUs, find one that works
			if (isDeviceSuitable(device)) 
			{         // If we find one or more, use it (this picks which one too)
				physicalDevice = device;
				break;
			}
		}
		if (physicalDevice == VK_NULL_HANDLE) 
		{
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		// Example
		//VkPhysicalDeviceProperties deviceProperties;        // Name, type, Version
		//vkGetPhysicalDeviceProperties(device, &deviceProperties);
		//VkPhysicalDeviceFeatures deviceFeatures;            // Texture compression, 64 bit floats, multi viewport rendering(VR)
		//vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		//return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
		//    deviceFeatures.geometryShader;

		QueueFamilyIndices indices = findQueueFamilies(device);     // Returns a struct of Queues
		return indices.isComplete();
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) 
	{
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr); // Takes device, count, container
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) 
		{						// From all the Queue families
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) 
			{			// If it has Queue graphics bit, 
				indices.graphicsFamily = i;			// The list is ordered and stored as ints, if it has this family, assign it index of Graphics bit so we know where it is
			}
			if (indices.isComplete()) {		// WE only need 1 so if sound, break;
				break;
			}
			i++;
		}
		return indices;
	}

	bool checkValidationLayerSupport() 
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) 	// For every predtermined layer (we have 1)
		{			
			bool layerFound = false;
			for (const auto& layerProperties : availableLayers) {				//  For every available layer
				if (strcmp(layerName, layerProperties.layerName) == 0) {		// If Available matches predetermiend
					layerFound = true;											// WE got her
					break;
				}
			}
			if (!layerFound) {		// Not found, cant use
				return false;
			}
		}
		return true;
	}

	void initWindow() 
	{
		glfwInit();  // Start it
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // Tell it its not OpenGL
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);     // Resize isn't straigth forward, set false
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // 4th is which monitor, 5th is GL(no use)
	}
	void mainLoop() 
	{
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
		}
	}

	void cleanup() 
	{
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}
};

int main() 
{
	HelloTriangleApplication app;

	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

// Example codes
//
//std::optional<uint32_t> graphicsFamily;
//
//std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl; // false
//
//graphicsFamily = 0;
//
//std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl; // true