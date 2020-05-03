#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

using namespace std;



/******************************************************/
/******************** Dataset Part ********************/
/******************************************************/

/* Split input string into vector strings */
vector<string> split(string input) {
	vector<string> result;

	// Split by tab
	for (int pos = input.find("\t"); pos != -1; pos = input.find("\t")) {
		result.push_back(input.substr(0, pos));
		input = input.substr(pos + 1);
	}

	// Erase enter
	int pos = input.find("\n");
	result.push_back(input.substr(0, pos));

	return result;
}

class Dataset {
public:
	/* Variable */
	vector<string> attrName;
	vector<map<string, int>> attrLabel;

	string className;
	map<string, int> classLabel;

	vector<vector<string>> attrData;
	vector<string> classData;



	/* Insert attribute name */
	void insertAttrName(vector<string> _attrName) {
		attrName = _attrName;

		attrLabel.resize(_attrName.size());
	}

	/* Insert class name */
	void insertClassName(string _className) {
		className = _className;
	}

	/* Insert attribute data */
	void insertAttrData(vector<string> _attrData) {
		attrData.push_back(_attrData);

		// Count the number of attributes labels
		for (int i = 0; i < _attrData.size(); i++) {
			auto it = attrLabel[i].find(_attrData[i]);

			if (it == attrLabel[i].end())
				attrLabel[i][_attrData[i]] = 1;
			else
				attrLabel[i][_attrData[i]]++;
		}
	}

	/* Insert class data */
	void insertClassData(string _classData) {
		classData.push_back(_classData);

		// Count the number of class labels
		auto it = classLabel.find(_classData);

		if (it == classLabel.end())
			classLabel[_classData] = 1;
		else
			classLabel[_classData]++;
	}



	/* Print attribute name */
	string printAttrName() {
		string result = "";

		for (int i = 0; i < attrName.size(); i++) {
			result.append(attrName[i]);
			result.append("\t");
		}

		return result;
	}

	/* Print class name */
	string printClassName() {
		string result = "";

		result.append(className);
		result.append("\n");

		return result;
	}

	/* Print attribute data */
	string printAttrData(int index) {
		string result = "";

		for (int i = 0; i < attrData[index].size(); i++) {
			result.append(attrData[index][i]);
			result.append("\t");
		}

		return result;
	}

	/* Print class data */
	string printClassData(int index) {
		string result = "";

		result.append(classData[index]);
		result.append("\n");

		return result;
	}
};



/************************************************************/
/******************** Decision Tree Part ********************/
/************************************************************/

struct Node {
	/* Variable */
	Dataset dataset;
	string datasetClass;
	int attr;

	map<string, Node> child;
};

class DecisionTree {
public:
	/* Variable */
	Node head;



	/* Decision tree */
	DecisionTree(Dataset train) {
		head.dataset = train;
		head.datasetClass = majorityVoting(head.dataset);
		head.attr = attributeSelection(head.dataset);

		makeChild(head);
	}



	/* Majority Voting */
	string majorityVoting(Dataset D) {
		int max = 0;
		string result;

		for (auto it = D.classLabel.begin(); it != D.classLabel.end(); it++) {
			if (max < it->second) {
				max = it->second;
				result = it->first;
			}
		}

		return result;
	}

	/* Attribute selection */
	int attributeSelection(Dataset D) {
		double max = 0;
		int result = -1;

		for (int i = 0; i < D.attrName.size(); i++) {
			double gainRatio = informationGain(D, i) / splitInformationAttribute(D, i);

			if (max < gainRatio) {
				max = gainRatio;
				result = i;
			}
		}

		return result;
	}

	/* Information gain */
	double informationGain(Dataset D, int A) {
		double result = information(D) - informationAttribute(D, A);

		return result;
	}

	/* Information */
	double information(Dataset D) {
		double result = 0;

		for (auto it = D.classLabel.begin(); it != D.classLabel.end(); it++) {
			double p = (double)it->second / (double)D.attrData.size();

			result -= p * log2(p);
		}

		return result;
	}

	/* Information attribute */
	double informationAttribute(Dataset D, int A) {
		double result = 0;

		for (auto it = D.attrLabel[A].begin(); it != D.attrLabel[A].end(); it++) {
			Dataset Da;

			Da.insertAttrName(D.attrName);
			Da.insertClassName(D.className);

			for (int i = 0; i < D.attrData.size(); i++) {
				if (!(it->first).compare(D.attrData[i][A])) {
					Da.insertAttrData(D.attrData[i]);
					Da.insertClassData(D.classData[i]);
				}
			}

			result += (double)Da.attrData.size() / (double)D.attrData.size() * information(Da);
		}

		return result;
	}

	/* Split information attribute */
	double splitInformationAttribute(Dataset D, int A) {
		double result = 0;

		for (auto it = D.attrLabel[A].begin(); it != D.attrLabel[A].end(); it++) {
			double p = (double)it->second / (double)D.attrData.size();

			result -= p * log2(p);
		}

		return result;
	}



	/* Make child */
	void makeChild(Node& node) {
		Dataset D = node.dataset;
		int A = node.attr;



		if (D.classLabel.size() == 1)
			return;
		if (D.attrName.empty())
			return;
		if (D.attrData.empty())
			return;



		for (auto it = D.attrLabel[A].begin(); it != D.attrLabel[A].end(); it++) {
			Dataset Da;

			vector<string> attrName = D.attrName;
			attrName.erase(attrName.begin() + A);

			Da.insertAttrName(attrName);
			Da.insertClassName(D.className);

			for (int i = 0; i < D.attrData.size(); i++) {
				if (!(it->first).compare(D.attrData[i][A])) {
					vector<string> attrData = D.attrData[i];
					attrData.erase(attrData.begin() + A);

					Da.insertAttrData(attrData);
					Da.insertClassData(D.classData[i]);
				}
			}



			Node temp;
			temp.dataset = Da;
			temp.datasetClass = majorityVoting(temp.dataset);
			temp.attr = attributeSelection(temp.dataset);
			node.child[it->first] = temp;

			makeChild(node.child[it->first]);
		}
	}



	/* Predict */
	Dataset predict(Dataset test) {
		Dataset result;

		result.insertAttrName(head.dataset.attrName);
		result.insertClassName(head.dataset.className);

		for (int i = 0; i < test.attrData.size(); i++) {
			result.insertAttrData(test.attrData[i]);
			result.insertClassData(DFS(head, test.attrData[i]));
		}

		return result;
	}

	/* DFS */
	string DFS(Node& node, vector<string> _attrData) {
		if (node.child.empty())
			return node.datasetClass;

		auto it = node.child.find(_attrData[node.attr]);
		if (it == node.child.end())
			return node.datasetClass;

		vector<string> attrData = _attrData;
		attrData.erase(attrData.begin() + node.attr);

		return DFS(it->second, attrData);
	}
};



/***************************************************/
/******************** Main Part ********************/
/***************************************************/

int main(int argc, char* argv[]) {
	/* Arguments */
	if (argc != 4) {
		cout << "Error : Arguments" << endl;

		return 0;
	}



	/* Train file */
	Dataset train;

	ifstream input;
	input.open(argv[1]);

	if (!input.is_open()) {
		cout << "Error : Train file" << endl;

		return 0;
	}

	while (!input.eof()) {
		string str;
		getline(input, str);
		vector<string> token = split(str);

		if (token.size() == 1)
			continue;

		// Insert attribute name and class name
		if (train.attrName.empty()) {
			vector<string> attrName = token;
			attrName.pop_back();
			string className = token.back();

			train.insertAttrName(attrName);
			train.insertClassName(className);
		}
		// Insert attribute data and class data
		else {
			vector<string> attrData = token;
			attrData.pop_back();
			string classData = token.back();

			train.insertAttrData(attrData);
			train.insertClassData(classData);
		}
	}

	input.close();



	/* Decision tree */
	DecisionTree dt(train);



	/* Test file */
	Dataset test;

	input.open(argv[2]);

	if (!input.is_open()) {
		cout << "Error : Test file" << endl;

		return 0;
	}

	while (!input.eof()) {
		string str;
		getline(input, str);
		vector<string> token = split(str);

		if (token.size() == 1)
			continue;

		// Insert attribute name
		if (test.attrName.empty()) {
			vector<string> attrName = token;

			test.insertAttrName(attrName);
		}
		// Insert attribute data
		else {
			vector<string> attrData = token;

			test.insertAttrData(attrData);
		}
	}

	input.close();



	/* Result file */
	Dataset result = dt.predict(test);

	ofstream output;
	output.open(argv[3]);

	if (!output.is_open()) {
		cout << "Error : Result file" << endl;

		return 0;
	}

	output << result.printAttrName();
	output << result.printClassName();
	for (int i = 0; i < result.attrData.size(); i++) {
		output << result.printAttrData(i);
		output << result.printClassData(i);
	}

	output.close();



	return 0;
}