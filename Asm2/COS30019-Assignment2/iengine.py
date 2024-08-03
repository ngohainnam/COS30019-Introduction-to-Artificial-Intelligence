
#---------------------------------------------------------------------------------------------------------------------
'''
Assignment 2- Inference Engine
Student Name 1: Hai Nam Ngo
Student ID 1: 103488515

Student Name 2: Aston Lynch 
Student ID 2: 103964552
'''
#---------------------------------------------------------------------------------------------------------------------
import sys  #this is used mainly for reading text file and handle data in it.
import os   #this is used for handling error related to path file.
from itertools import product
from tabulate import tabulate
import re
#--------------------------------------------------------------------------------------------------------------------- 
class TextFileAnalyzer:   
    def __init__(self, filename):    
        self.filename = filename

    #Update: I moved the check validation part in main class to here.    
    def read_file(filename):
        if len(sys.argv) != 3:
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: You need to use the following format "python iengine.py <filename> <method>". Please check the command.\n')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()  
        
        else:
            if not os.path.exists(filename):
                print('-----------------------------------------------------------------------------------------------------------------------\n')
                print('Error: The map file does not exist. Please check the file path.\n')
                print('-----------------------------------------------------------------------------------------------------------------------')
                sys.exit()  


        #This class works perfectly now
        #To test it, use this command: python .\iengine.py .\test_HornKB.txt TT
        #This is the logic:
        #1. Open the file
        #2. Locate the position of 'TELL' and 'ASK' keywords in the content
        #3. Extract the knowledge base (KB) portion after 'TELL' and before 'ASK'
        #4. Extract the query portion after 'ASK'
        #5. Split the KB content into individual clauses based on ';' and strip each clause
        #6. Return the extracted knowledge base, query
        #Status: Working well    
                        
        with open(filename, 'r') as file:
            content = file.read()
        
        tell_index = content.find('TELL')
        ask_index = content.find('ASK')

        if tell_index == -1 or ask_index == -1:
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: The file content is not formatted correctly. Please check the file content.')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()
        
        kb_content = content[tell_index + 4:ask_index].strip()
        query = content[ask_index + 3:].strip()

        if not kb_content or not query:
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: The file content is not formatted correctly. Please check the file content.')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()
        
        kb = [clause.strip() for clause in kb_content.split(';') if clause.strip()]
           
        #Check if the text file is generic or not, if yes, only continue with TT method.
        if (sys.argv[2].lower() not in ["tt", "dpll"]):
            Additional_Connectives = ['<=>', '||', '~']
            symbol_set = set()
            for clause in kb:
                # Find additional connectives in each clause
                for connective in Additional_Connectives:
                    if connective in clause:
                        symbol_set.add(connective)
                           
            # If any additional connectives are found, print an error and exit
            if symbol_set:
                print('-----------------------------------------------------------------------------------------------------------------------')
                print('Error: Only TT/DPLL method can read generic KB with additional connectives: ', ', '.join(symbol_set))
                print('-----------------------------------------------------------------------------------------------------------------------')
                sys.exit()
            
        return kb, query
    
#--------------------------------------------------------------------------------------------------------------------- 
class TT:
    def __init__(self, kb, query):    
        self.model_list = []
        self.kb = kb
        self.query = query

    '''
    This function will create a list of symbols from the KB and pass the list to other subclasses
    Status: Working well
    '''   
    def ExtractSymbols(self, kb):
        pattern = r'[a-zA-Z]+\d*'
        symbol_set = set() 
        for clause in kb:
            symbols = re.findall(pattern, clause)
            symbol_set.update(symbols)  
        return list(symbol_set)  

    '''
    The logic of this class comes from the 'Inference by Enumeration' slide in Week 7
    Status: 
        Entails works
        CheckAll is still in progress
    '''   
    def CheckEntails(self):
        #Visualize the table with valid models
        valid_model_count = self.CreateTruthTable()
        
        if valid_model_count:
            print('YES:', valid_model_count)
        else:
            print('NO')

    # This one is mainly for visualization, using tabulate python library (to create table based on existing data)
    def CreateTruthTable(self):
        symbols = self.ExtractSymbols(self.kb)
        models = list(product([True, False], repeat=len(symbols)))

        data = []
        headers = symbols + self.kb + [self.query]

        valid_model_count = 0  # Initialize valid model count
        
        for model_values in models:
            model = dict(zip(symbols, model_values))
            row = [model[symbol] for symbol in symbols]
            
            kb_values = [self.Check_if_clause_true(clause, model) for clause in self.kb]
            row.extend(kb_values)
            
            query_value = self.Check_if_clause_true(self.query, model)
            row.append(query_value)

            # Apply coloring to valid models and count them
            if all(kb_values) and query_value:
                valid_model_count += 1
                GREEN = '\033[92m'
                END = '\033[0m'
                colored_row = [f"{GREEN}{str(cell)}{END}" for cell in row]
            else:
                colored_row = [f"{str(cell)}" for cell in row]

            data.append(colored_row)

        # Print the table with valid models
        print(tabulate(data, headers=headers, tablefmt="fancy_grid"))
        return valid_model_count
        
    '''
    The purpose of this function:
    Determines if the entire KB is satisfied by a given model
    Uses the is_true function to check each clause
    
    Expected output:
        True if all clauses in the KB are satisfied
        False if any clause in the KB is not satisfied
    '''   
    def Check_if_kb_true(self, model):
        for clause in self.kb:
            if not self.Check_if_clause_true(clause, model):
                return False
        return True

    '''
    The purpose of this function:
    Evaluates a single clause based on the truth values provided in the model
    Handles both Horn clauses (with implications =>) and single symbol clauses
    
    Expected output: 
        True if the clause is satisfied according to the model
        False otherwise
    '''
    def Check_if_clause_true(self, clause, model):
        clause = clause.replace(" ", "")
        return self.EvaluateClause(clause, model)

    '''
    The purpose of this function:
    Evaluates a given logical clause based on the truth values provided in the model
    Handles conjunction, disjunction, negation, implication, and biconditional operators
    
    Expected output:
        True if the clause is satisfied according to the model
        False otherwise
    '''
    def EvaluateClause(self, clause, model):
        #Find the main logical operator in the clause and split it accordingly
        left,op,right = self.FindMainOperator(clause)
        
        #Evaluate the clause based on the identified main operator
        if op == '<=>':
            #Biconditional: true if both sides are equal
            return self.EvaluateClause(left, model) == self.EvaluateClause(right, model)
        
        elif op == '=>':
            #Implication: true if left side is false or right side is true
            return not self.EvaluateClause(left, model) or self.EvaluateClause(right, model)
        
        elif op == '||':
            #Disjunction: true if either side is true
            return self.EvaluateClause(left, model) or self.EvaluateClause(right, model)
        
        elif op == '&':
            #Conjunction: true if both sides are true
            return self.EvaluateClause(left, model) and self.EvaluateClause(right, model)
        
        elif clause.startswith('~'):
            #Negation: true if the negated clause is false
            return not self.EvaluateClause(clause[1:], model)
        
        else:
            #Single symbol: return its value in the model
            return model.get(clause.strip(), False)

    '''
    The purpose of this function:
    Identifies the main logical operator at the top level of the clause, considering nested parentheses
    
    Expected output:
        Returns the operator and the left and right parts of the clause split by the operator
    '''
    def FindMainOperator(self, clause):
        bracket_level = 0
        result = None
        
        for i in range(len(clause)):
            if clause[i] == '(':
                bracket_level += 1
                
            elif clause[i] == ')':
                bracket_level -= 1
                
            elif bracket_level == 0:  # Only consider operators at the top level
                if clause[i:i+3] == '<=>':
                    result = clause[:i], '<=>', clause[i+3:]
                    break
                elif clause[i:i+2] == '=>':
                    result = clause[:i], '=>', clause[i+2:]
                    break
                elif clause[i:i+2] == '||':
                    result = clause[:i], '||', clause[i+2:]
                    break
                elif clause[i] == '&':
                    result = clause[:i], '&', clause[i+1:]
                    break
        
        # I added this code block to remove bracket level that is not needed. Like this: (a & (b=>c)) will just return a & (b=>c)
        if result:
            left, op, right = result
            
            if left[0] == '(' and left[-1] == ')':
                left = left[1:-1]
                
            if right[0] == '(' and right[-1] == ')':
                right = right[1:-1]

            return left, op, right
        else:
            return clause, None, None

#---------------------------------------------------------------------------------------------------------------------
class Chaining:
    def __init__(self, kb,query): 
        self.kb = kb
        self.query = query

    '''
    The purpose of this function: finding the single clause (symbol) that is already true in the kb
    '''    
    def FindSingleClause(self):
        for clause in self.kb:
            if '=>' not in clause:
                self.queue.append(clause)
        return self.queue

    '''
    This section is made in order to convert the pure kb into sentence list (list of dictionary that contains information about conclusion, premises
    and length of the premises)
    '''         
    def GenerateSentenceList(self):
        for clause in self.kb:
            if '=>' in clause:
                premise, conclusion = clause.split('=>')        
                premise_symbols = [symbol.strip() for symbol in premise.split('&')]    
                conclusion = conclusion.strip()                 
                sentence = {'premise':premise_symbols,'conclusion':conclusion,'count':len(premise_symbols)}    
                self.sentence_list.append(sentence)    
        return self.sentence_list 
    
#---------------------------------------------------------------------------------------------------------------------
class FC(Chaining):
    def __init__(self, kb,query):  
        super().__init__(kb,query)

        '''
        queue: this is for storing single clause (symbol) that is already true in the kb
        sentence_list: this is for storing the list of sentence (dictionary for each clause, with split conclusion and premises)
        inferred: this is for storing the symbol in the queue that is already analyzed (help us to create the final output)
        '''
        self.queue = []
        self.sentence_list = []
        self.inferred = []

    '''
    the purpose of this function: Check for entailment
    '''
    def CheckEntails(self):
        self.FindSingleClause()
        self.GenerateSentenceList()
        
        '''
        the logic of this section:
        1. pop out the current symbol
        2. add it to the inferred list (since it is analyzed)
        3. loop through each sentence (dictionary) in the sentence list (dictionary list)
        4. if the symbol is inside the sentence's premise, it will reduce the length of the premise by 1 
        (this means that if since that symbol is true, then we don't need to care about it anymore, we just need to check for other symbols)
        
        when the length of the premise is down to 0, that means every symbol in the premise is true, which means the conclusion is true
        then we can add it to the queue to continue analyze, until we reach the query
        '''
        while self.queue:
            current_symbol = self.queue.pop(0)
            self.inferred.append(current_symbol)
            
            for sentence in self.sentence_list:
                if current_symbol in sentence['premise']:
                    sentence['count'] = sentence.get('count') - 1
                    
                    if sentence['count'] == 0:
                        self.queue.append(sentence['conclusion'])
        
            '''
            This section is made for displaying output
            if the query is in the inferred list, that means we have reached the goal, now we just need to print out the result
            '''    
            if self.query in self.inferred:
                print('YES:', ', '.join(self.inferred))
                return True
            
        #Otherwise, the output will be NO   
        print('NO')
        return False
    
#---------------------------------------------------------------------------------------------------------------------
class BC(Chaining):
    def __init__(self, kb,query):  
        super().__init__(kb,query)

        '''
        queue: this is for storing single clause (symbol) that is already true in the kb
        sentence_list: this is for storing the list of sentence (dictionary for each clause, with split conclusion and premises)
        inferred: this is for storing the symbol in the queue that is already analyzed (help us to create the final output)
        '''
        self.queue = []
        self.sentence_list = []
        self.inferred = []
        self.visited = set()

    '''
    the purpose of this function: Check for entailment
    the logic is inspired from the assignment 2 helper on Canvas
    '''
    def CheckEntails(self):
        self.FindSingleClause()
        self.GenerateSentenceList()
        
        if self.TruthValue(self.query):
            print('YES:', ', '.join(self.inferred))
            return True
        else:
            print('NO')
            return False

    '''
    The purpose of this function is check the right-hand side (conclusion) to determine if it is true or not, by checking the truth value 
    of every premises (the left-hand side)
    '''
    def TruthValue(self, symbol):
        if symbol in self.inferred:
            return True
        
        if symbol in self.queue:
            self.inferred.append(symbol)
            return True
        
        if symbol in self.visited:
            return False

        self.visited.add(symbol)

        for sentence in self.sentence_list:
            if symbol == sentence['conclusion']:
                all_premises_true = all(self.TruthValue(premise) for premise in sentence['premise'])
                if all_premises_true and symbol not in self.inferred:
                    self.queue.append(symbol)
                    self.inferred.append(symbol)
                    return True

        return False

#---------------------------------------------------------------------------------------------------------------------     
class DPLL:
    def __init__(self, kb, query):
        self.kb = self.convert_to_clauses(kb)
        self.query = query

    def CheckEntails(self):
        """
        Check if the query is entailed by the knowledge base using the DPLL algorithm.
        """
        result, model_count = self.dpll(self.kb, {})
        if result:
            print(f'YES: {model_count}')
        else:
            print('NO')

    def convert_to_clauses(self, kb):
        """
        Convert the knowledge base into a list of clauses.
        :return: List of lists, where each inner list represents a clause.
        """
        clauses = []
        for clause in kb:
            clauses.append(self.parse_clause(clause))
        return clauses

    def parse_clause(self, clause):
        """
        Convert a string clause into a list of literals.

        :param clause: String representing a clause.
        :return: List of strings, where each string is a literal in the clause.
        """
        literals = clause.split('|')
        return [literal.strip() for literal in literals]

    def dpll(self, clauses, assignment):
        """
        Recursive DPLL algorithm 
        :param clauses: List of lists, where each inner list represents a clause.
        :param assignment: Dictionary representing the current variable assignments.
        :return: Tuple (boolean, int) where the boolean indicates if the clauses are satisfiable,
                 and the int represents the number of valid models.
        """
        # Base cases
        if all(self.evaluate_clause(clause, assignment) for clause in clauses):
            return True, 1
        if any(self.evaluate_clause(clause, assignment) == False for clause in clauses):
            return False, 0

        # Unit propagation
        unit_clauses = [clause for clause in clauses if len(clause) == 1]
        if unit_clauses:
            unit = unit_clauses[0][0]
            return self.dpll(self.simplify_clauses(clauses, unit), {**assignment, unit: True})

        # Pure literal elimination
        literals = {literal for clause in clauses for literal in clause}
        pure_literals = {literal for literal in literals if f'-{literal}' not in literals and literal[1:] not in literals}
        if pure_literals:
            pure_literal = next(iter(pure_literals))
            return self.dpll(self.simplify_clauses(clauses, pure_literal), {**assignment, pure_literal: True})

        # Splitting
        literal = next(iter(literals))
        result_true, count_true = self.dpll(self.simplify_clauses(clauses, literal), {**assignment, literal: True})
        result_false, count_false = self.dpll(self.simplify_clauses(clauses, f'-{literal}'), {**assignment, literal: False})
        return (result_true or result_false), (count_true + count_false)

    def simplify_clauses(self, clauses, literal):
        """
        Simplify the clauses by removing satisfied clauses and literals.

        :param clauses: List of lists, where each inner list represents a clause.
        :param literal: String representing the literal to simplify by.
        :return: Simplified list of clauses.
        """
        simplified_clauses = []
        for clause in clauses:
            if literal in clause:
                continue
            simplified_clause = [lit for lit in clause if lit != f'-{literal}' and lit != literal[1:]]
            simplified_clauses.append(simplified_clause)
        return simplified_clauses

    def evaluate_clause(self, clause, assignment):
        """
        Evaluate a single clause based on the current assignment.

        :param clause: List of strings representing the literals in a clause.
        :param assignment: Dictionary representing the current variable assignments.
        :return: True if the clause is satisfied, False if it is falsified, None if undetermined.
        """
        for literal in clause:
            if literal in assignment and assignment[literal] == True:
                return True
            if f'-{literal}' in assignment and assignment[f'-{literal}'] == False:
                return True
        return None

#---------------------------------------------------------------------------------------------------------------------        
class Main:            
        filename = sys.argv[1]
        method = sys.argv[2] 
        print(sys.argv[1], sys.argv[2]) 
        
        result = TextFileAnalyzer.read_file(filename)
        kb = result[0]
        query = result[1]
        print ('This is the KB: ',kb)
        print ('This is the QUERY: ',query)

        #Choose the appropriate method based on the input
        if method.lower() == 'fc':
            algorithm = FC(kb, query)                         
        elif method.lower() == 'bc':
            algorithm = BC(kb, query)            
        elif method.lower() == 'tt':
            algorithm = TT(kb, query)
        elif method.lower() == 'dpll':
            algorithm = DPLL(kb, query)
        else:
            #Handle incorrect method input
            print('-----------------------------------------------------------------------------------------------------------------------\n')
            print('Error: Wrong search method input. Please check the command. \n')
            print('-----------------------------------------------------------------------------------------------------------------------')
            sys.exit()  #Exit the program if the method input is incorrect
            
        algorithm.CheckEntails()    
        sys.exit()
            