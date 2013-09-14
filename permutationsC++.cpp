#include <iostream>
#include <vector>
#include <string>

using namespace std;


vector<string> permute(string str) {
	vector<string> result;
	if (str.size() == 0) {
		result.push_back("");
		return result;
	}

}

int main() {
	cout << "enter a word to permute: ";
	string str;
	cin >> str;
	vector<string> vec = permute(str);
	for (vector<string>::iterator itr = vec.begin(); itr != vec.end(); itr++) {
		cout << *itr << endl;
	}
	return 0;
}

