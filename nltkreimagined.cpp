#include <iostream>
#include <string>
#include <vector>

// Tokenizer
class Tokenizer {
public:
  std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string::size_type pos = 0;
    while (pos != std::string::npos) {
      auto word_start = text.find_first_not_of(" \t\n\r", pos);
      auto word_end = text.find_first_of(" \t\n\r", word_start); 
      if (word_start != std::string::npos) {
        std::string token = text.substr(word_start, word_end - word_start);
        tokens.push_back(token);
      }
      pos = word_end;
    }
    return tokens;
  }
};

// Stemmer 
class Stemmer {
public:
  std::string stem(const std::string& word) {
    // Simple suffix stripping stemmer
    std::string stemmed = word;
    if (word.size() >= 3) {
      if (word.substr(word.size()-3) == "ing") {
        stemmed = word.substr(0, word.size()-3);  
      }
      if (word.substr(word.size()-2) == "es") {
        stemmed = word.substr(0, word.size()-2);
      }
    }
    return stemmed;
  }
};

int main() {
  
  Tokenizer tokenizer;
  Stemmer stemmer;
  
  std::string text = "These words are being tokenized and stemmed.";
  
  auto tokens = tokenizer.tokenize(text);
  
  for (const auto& token : tokens) {
    std::string stemmed = stemmer.stem(token);  
    std::cout << token << " => " << stemmed << std::endl;
  }

  return 0;
}

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

// Tokenizer 
class Tokenizer {
  // Same as before...
}; 

// Stemmer
class Stemmer {
  // Same as before ...
};

// Part-of-Speech Tagger
class PosTagger {
public:
  PosTagger() {
    // Initialize simple default tagged word mappings
    tagged_words["time"] = "NOUN"; 
    tagged_words["year"] = "NOUN";
    tagged_words["day"] = "NOUN";
    // ...
  }

  std::string tag(const std::string& word) {
    // Return stored tag if exists, otherwise return NOUN as default
    return tagged_words.count(word) > 0 ? tagged_words[word] : "NOUN";  
  }

private:
  std::unordered_map<std::string, std::string> tagged_words;  
};

// Named Entity Recognizer 
class Ner {  
public:
  std::string recognize(const std::string& word) {
    if (places.count(word) > 0) {
      return "GPE"; // Geo-political entity 
    }
    if (organizations.count(word) > 0) { 
      return "ORG";  
    }
    
    return "O"; // Other 
  }

private:
  std::unordered_map<std::string, int> places {
    {"France", 0}, {"Germany", 0}, {"Madrid", 0}
  };

  std::unordered_map<std::string, int> organizations {
    {"Microsoft", 0}, {"Apple", 0}, {"IBM", 0}  
  };  
};

int main() {

  Tokenizer tokenizer;  
  Stemmer stemmer;
  PosTagger pos_tagger;
  Ner ner;
  
  // Text analysis example
  std::string text = "Apple announced new products at a Microsoft event in Madrid, Spain this year.";  
  
  auto tokens = tokenizer.tokenize(text);
  
  for (const auto& token : tokens) {
   
    std::string stemmed = stemmer.stem(token);
    std::string tag = pos_tagger.tag(token);
    std::string entity = ner.recognize(token);
    
    std::cout << token << " " 
              << "[STEMMED: " << stemmed << "]"
              << "[POS: " << tag << "]"  
              << "[ENTITY: " << entity << "]" << std::endl;
  }

  return 0;
}

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

// Tokenization interfaces
class Tokenizer {
public:
  virtual std::vector<std::string> tokenize(const std::string& text) = 0;
};

class WordTokenizer : public Tokenizer {
public:
  std::vector<std::string> tokenize(const std::string& text) override; 
};

class SentenceTokenizer : public Tokenizer {  
public:
  std::vector<std::string> tokenize(const std::string& text) override;
};

// Stemming interfaces
class Stemmer {  
public:
  virtual std::string stem(const std::string& word) = 0;
};

class PorterStemmer : public Stemmer {
public:
  std::string stem(const std::string& word) override;  
};

// Part-of-Speech tagging interfaces
class PosTagger {
public:
  virtual std::string tag(const std::string& token) = 0; 
};

class DefaultPosTagger : public PosTagger {
public:
  std::string tag(const std::string& token) override;
};

class NgramPosTagger : public PosTagger {  
public:
  std::string tag(const std::string& token) override;
};

// Named Entity Recognition interfaces
class Ner {
public:
  virtual std::string recognize(const std::string& token) = 0;  
};

class DictionaryNer : public Ner {
public:
  std::string recognize(const std::string& token) override;  
};


// Corpus & Vocabulary 
class Corpus {
public:
  std::vector<std::string> documents;  
};

class Vocabulary {
public: 
  std::unordered_map<std::string, int> term_counts; 
};

// Analysis Pipeline  
class Pipeline {
public:
  void add(Tokenizer* tokenizer);
  void add(Stemmer* stemmer);
  void add(PosTagger* pos_tagger);
  void add(Ner* ner);
  
  std::vector<std::string> analyze(const std::string& text);

private:
  std::vector<Tokenizer*> tokenizers;
  std::vector<Stemmer*> stemmers;  
  std::vector<PosTagger*> pos_taggers;
  std::vector<Ner*> ners;  
};

int main() {
  
  // Build pipeline
  Pipeline pipeline;
  pipeline.add(new WordTokenizer()); 
  pipeline.add(new PorterStemmer());
  pipeline.add(new DefaultPosTagger());
  
  // Run text analysis
  auto results = pipeline.analyze("Sample text for analysis...");  

  return 0;
}

