package twoLayerNN;

import java.util.Arrays;

public class NeuralNetwork {
	
	/*
	 * activation function: logistic function
	 * f(x) = 1 / (1 + e^(-x))
	 * 
	 * NOTE: WE DO NOT COUNT 'INPUTS' AS A LAYER
	 * 
	 * size_list = { num_of_neurons_in_layer_0 (inputs),
	 * 			     num_in_layer_1 (first hidden layer),
	 * 			 	 ...,
	 * 			 	 num_in_last_layer (outputs) }
	 * 
	 * num_inputs = size_list[0]
	 * num_layers = size_list.length - 1 = sizes.length
	 * sizes = { num_in_layer_1 (first hidden layer),
	 * 			 ...,
	 * 			 num_in_last_layer (outputs) }
	 * 
	 * z = net input = weighted input
	 * x = inputs = activations of prev layer = outputs of prev layer
	 * a = outputs = activations of current layer = inputs of next layer
	 * y = target outputs
	 * cost function = loss function = objective = error function
	 * activation function = logistic function = sigmoid function
	 */
	final double LEARNING_RATE = 0.5;
	int num_layers;
	int num_inputs;
	int[] sizes;
	NeuronLayer[] layers;
	
	public NeuralNetwork (int[] size_list) {
		this.num_layers = size_list.length - 1;
		this.sizes = Arrays.copyOfRange(size_list, 1, size_list.length);
		this.num_inputs = size_list[0];
		
		// randomize values for weights and biases
		double[][] weights = new double[this.num_layers][];
		for (int i = 0; i < this.num_layers; i++) {
			int num_weights_for_layer = size_list[i]*size_list[i+1];
			weights[i] = randomize_array(num_weights_for_layer);
		}
		double[] biases = randomize_array(this.num_layers);
		
		create_layers(weights, biases);
//		this.inspect();
	}
	
	public NeuralNetwork (int[] size_list, double[][] weights, double[] biases) {
		this.num_layers = size_list.length - 1;
		this.sizes = Arrays.copyOfRange(size_list, 1, size_list.length);
		this.num_inputs = size_list[0];
		
		create_layers(weights, biases);
	}
	
	public void create_layers (double[][] weights, double[] biases) {
		this.layers = new NeuronLayer[this.num_layers];
		for(int i = 0; i < this.num_layers; i++) {
			this.layers[i] = new NeuronLayer(this.sizes[i], weights[i], biases[i]);
		}
	}
	
	public double[] randomize_array (int size) {
		double[] rands = new double[size];
		for (int i = 0; i < size; i++) {
			rands[i] = Math.random();
		}
		return rands;
	}
	
	
	
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	public void train (double[] inputs, double[] target_outputs) {
		// activations' first row = inputs
		double[][] activations = feedforward(inputs);
//		print_array("activations from feedforward", activations);
		
		
		double[] outputs = activations[activations.length-1];
//		print_array("outputs from feedforward", outputs);
		
		
		double[] errors = total_error(outputs, target_outputs);
		print_err(errors);
		
		
		double[] output_deltas = get_output_deltas(outputs, target_outputs);
//		print_array("output deltas", output_deltas);
		
		
		double[][] deltas = new double[this.num_layers][];
		deltas[deltas.length-1] = output_deltas;
		for (int i = deltas.length - 2; i >= 0; i--) {
			deltas[i] = layers[i+1].get_layer_deltas(deltas[i+1], activations[i+1]);
//			print_array("activations[" + (i+1) + "]", activations[i+1]);
		}
//		print_array("deltas", deltas);
		
		
		double[][][] gradients_w = new double[this.num_layers][][];
		for (int i = 0; i < gradients_w.length; i++) {
			// the reason the inputs here aren't like 'i-1' and 'i', is bc our
			// delta matrices only have 2 layers, we foregoed the 'input' layer
			gradients_w[i] = layers[i].get_gradients_w(activations[i], deltas[i]);
		}
//		print_array("gradients (WEIGHTS)", gradients_w);
		
		
		double[][] gradients_b = new double[this.num_layers][];
		for (int i = 0; i < gradients_b.length; i++) {
			gradients_b[i] = layers[i].get_gradients_b(deltas[i]);
		}
//		print_array("gradients (BIASES)", gradients_b);
		
		
//		this.inspect();
		this.update_network(gradients_w, gradients_b);
//		this.inspect();
	}
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	
	
	
	public double[][] feedforward (double[] inputs) {
		// activations[i] = outputs of layer i, inputs of layer i+1
		double[][] activations = new double[this.num_layers+1][];
		activations[0] = inputs;
		
		for (int i = 0; i < this.num_layers; i++) {
			activations[i+1] = this.layers[i].feedforward(activations[i]);
		}
		return activations;
	}
	
	public double[] total_error(double[] outputs, double[] target_outputs) {
		int n = (outputs.length == target_outputs.length) ? outputs.length : -1;
		if (n < 0) throw new Error("outputs and targets' lengths do not match!!");
		
		double[] errors = new double[n];
		for (int i = 0; i < n; i++) {
			errors[i] = cost_function(outputs[i], target_outputs[i]);
		}
		
		return errors;
	}
	
	public double cost_function (double output, double target) {
		return 0.5 * Math.pow(target - output, 2);
	}
	
	// EQUATION 1 (BP1)
	// δ^L_j = (∂C / ∂a^L_j) * σ′(z^L_j)
	// 		 = [-(target - output)] * [output(1-output)]
	//		 = (a - y) * a(1 - a)
	
	// from δ = (∂E / ∂z)
	//		  = (∂C / ∂a)
	// 		  = (∂E / ∂a) * (∂a / ∂z)
	//		  = (∂C / ∂a^L_j) * σ′(z^L_j)     etc
	public double[] get_output_deltas (double[] outputs, double[] targets) {
		double[] deltas = new double[outputs.length];
		for (int i = 0; i < outputs.length; i++) {
			deltas[i] = cost_derivative(outputs[i], targets[i]) * activate_derivative(outputs[i]);
		}
		return deltas;
	}
	
	public double cost_derivative (double output, double target) { // with respect to output/activation
		return -(target - output);
	}
	
	public static double activate (double z) {
		return 1 / (1 + Math.exp(-z));
	}
	
	public static double activate_derivative (double output) {
		return output * (1 - output);
	}
	
	public void update_network (double[][][] gradients_w, double[][] gradients_b) {
		for (int i = 0; i < this.num_layers; i++) {
			layers[i].update_network_layer(gradients_w[i], gradients_b[i], this.LEARNING_RATE);
		}
	}
	
	public void try_inputs (double[] inputs, double[] expected) {
		double[] outputs = feedforward(inputs)[this.num_layers];
		System.out.println("inputs: " + Arrays.toString(inputs));
		System.out.println("expected: " + Arrays.toString(expected));
		System.out.println("outputs: " + Arrays.toString(outputs) + "\n");
	}
	
	public void inspect() {
		System.out.println("\t----------------------------------------------------------------------------------");
		for (int i = 0; i < layers.length; i++) {
			layers[i].inspect();
			System.out.println("\t----------------------------------------------------------------------------------");
		}
	}
	
	public static void print_array(String title, double[] arr) {
		System.out.println(title + ":");
		System.out.println("\t" + Arrays.toString(arr));
		System.out.println();
	}
	
	public static void print_array(String title, double[][] arr) {
		System.out.println(title + ":");
		for (int i = 0; i < arr.length; i++)
			System.out.println("\t" + Arrays.toString(arr[i]));
		System.out.println();
	}
	
	public static void print_array(String title, double[][][] arr) {
		System.out.println(title + ":");
		for (int i = 0; i < arr.length; i++)
			for (int j = 0; j < arr[i].length; j++)
			System.out.println("\t" + Arrays.toString(arr[i][j]));
		System.out.println();
	}
	
	public static void print_err(double[] err) {
//		the multiplication is to scale the error so it's easier to compare when we have many iterations
//		for (int i = 0; i < err.length; i++) err[i] *= 100;
		System.out.print("errors: " + Arrays.toString(err));
		System.out.println();
	}
	
	public static void main (String[] args) {
//		given initial values testing example
//		int[] size_list = {2, 2, 2};
//		double[] biases = {0.35, 0.60};
//		double[][] weights = {
//								{0.15, 0.20, 0.25, 0.30},
//								{0.40, 0.45, 0.50, 0.55}
//							 };
//		double[] inputs = {0.05, 0.10};
//		double[] target_outputs = {0.01, 0.99};
//		
//		NeuralNetwork nn = new NeuralNetwork(size_list, weights, biases);
//		nn.train(inputs, target_outputs);
		
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		
//		random initial values testing example
//		NeuralNetwork randnn = new NeuralNetwork(size_list);
//		randnn.train(inputs, target_outputs);
		
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		
//		XOR function testing example:
		double[][][] training_sets = {
				{{0, 0}, {0}},		// form: {{inputs}, {expected output}}
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
		};
		int[] size_list = {training_sets[0][0].length, 5, training_sets[0][1].length};
		NeuralNetwork XORnn = new NeuralNetwork(size_list);
		for (int i = 0; i < 1000000; i++) {
			int rand_choice = (int) Math.floor(Math.random()*training_sets.length);
			XORnn.train(training_sets[rand_choice][0], training_sets[rand_choice][1]);
		}
		
		System.out.println("\n------------------------------------------------------\n");
		for (int i = 0; i < training_sets.length; i++)
			XORnn.try_inputs(training_sets[i][0], training_sets[i][1]);
		System.out.println("------------------------------------------------------");
		
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		
//		AND function testing example:
//		double[][][] training_sets = {
//				{{0, 0}, {0}},		// form: {{inputs}, {expected output}}
//				{{0, 1}, {0}},
//				{{1, 0}, {0}},
//				{{1, 1}, {1}},
//		};
//		int[] size_list = {training_sets[0][0].length, 5, training_sets[0][1].length};
//		NeuralNetwork ANDnn = new NeuralNetwork(size_list);
//		for (int i = 0; i < 1000000; i++) {
//			int rand_choice = (int) Math.floor(Math.random()*training_sets.length);
//			ANDnn.train(training_sets[rand_choice][0], training_sets[rand_choice][1]);
//		}
//		
//		System.out.println("\n------------------------------------------------------\n");
//		for (int i = 0; i < training_sets.length; i++)
//			ANDnn.try_inputs(training_sets[i][0], training_sets[i][1]);
//		System.out.println("------------------------------------------------------");
		
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		
//		OR function testing example:
//		double[][][] training_sets = {
//				{{0, 0}, {0}},		// form: {{inputs}, {expected output}}
//				{{0, 1}, {1}},
//				{{1, 0}, {1}},
//				{{1, 1}, {1}},
//		};
//		int[] size_list = {training_sets[0][0].length, 5, training_sets[0][1].length};
//		NeuralNetwork ORnn = new NeuralNetwork(size_list);
//		for (int i = 0; i < 1000000; i++) {
//			int rand_choice = (int) Math.floor(Math.random()*training_sets.length);
//			ORnn.train(training_sets[rand_choice][0], training_sets[rand_choice][1]);
//		}
//		
//		System.out.println("\n------------------------------------------------------\n");
//		for (int i = 0; i < training_sets.length; i++)
//			ORnn.try_inputs(training_sets[i][0], training_sets[i][1]);
//		System.out.println("------------------------------------------------------");
	}
}