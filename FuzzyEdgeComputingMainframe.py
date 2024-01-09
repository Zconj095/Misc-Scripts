import fuzzylite as fl
from edgecompute import EdgeComputer

class FuzzyEdgeMainframe:
    def __init__(self):
        self.fuzzy_engine = fl.Engine()
        self.edge_devices = []
        
    def connect_device(self, device):
        self.edge_devices.append(device)
        device.mainframe = self
        
    def define_fuzzy_sets(self, data):
        # Analyze data ranges to define fuzzy membership functions 
        min_vals = [min(device.data) for device in self.edge_devices]
        max_vals = [max(device.data) for device in self.edge_devices]
        
        input_names = [device.input_name for device in self.edge_devices]
        
        mf_defs = []
        for i, input_name in enumerate(input_names):
            mf_low = fl.TrapezoidalMF(min_vals[i], min_vals[i], min_vals[i]*1.1, min_vals[i]*1.25)  
            mf_mid = fl.TriangularMF(min_vals[i]*1.25, (min_vals[i] + max_vals[i])/2, max_vals[i]*0.75)
            mf_high = fl.TrapezoidalMF(max_vals[i]*0.75, max_vals[i]*0.9, max_vals[i], max_vals[i])
            mf_defs.append((input_name, [mf_low, mf_mid, mf_high]))
            
        self.fuzzy_engine.add_input_variables([input_name for input_name, _ in mf_defs])  
        self.fuzzy_engine.add_output_variable("average_output")
        
        for input_name, mf_list in mf_defs:
            self.fuzzy_engine.add_membership_functions(mf_list, input_name)
            
    def process_and_analyze(self):
        inputs = {}
        
        for device in self.edge_devices:
            # Get fuzzified inputs 
            fuzzified_input = self.fuzzy_engine.fuzzify(device.input_name, device.get_new_data())  
            
            inputs[device.input_name] = fuzzified_input
            
            # Device processes data 
            output = device.process_data(fuzzified_input)  
            
            # Send output to the cloud for analysis
            send_to_cloud(device.device_id, output)  
            
        # Fuzzy inference on edge devices outputs
        inferred_output = self.fuzzy_engine.infer(inputs)
        
        defuzzified = self.fuzzy_engine.defuzzify("average_output", inferred_output, 'centroid') 
        
        # Further analytics on fuzzy edge output
        analyze_fuzzy_output(defuzzified)   
        
class EdgeComputer:
    # Class for edge device
    def __init__(self, device_id, input_name):
        self.mainframe = None  
        self.device_id = device_id
        self.input_name = input_name
        self.data = [] 
        
    def get_new_data(self):
        new_data = # get new data 
        self.data.append(new_data)
        return new_data
        
    def process_data(self, input):
        # Process data using fuzzy input  
        output = # process fuzzified input 
        return output
