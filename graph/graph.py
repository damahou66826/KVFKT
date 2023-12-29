import edgeNode



class graph(object):
    def  __init__(self):
        self.questionNodes = list()
        self.studentNodes = list()
        self.edgeNodes = list()
        self.question4student = dict()
        self.student4question = dict()

    def add(self, studentId, questionId, conceptId, timestamp, value_memory_matrix):
        questionIdSet = list()
        if(studentId in self.student4question.keys()):
            questionIdSet = self.student4question.get(studentId)

        studentIdSet = list()
        if(questionId in self.question4student.keys()):
            studentIdSet = self.question4student.get(questionId)

        #向图中插入该条边
        tempEdge = edgeNode(studentId, questionId, conceptId, timestamp, value_memory_matrix)
        self.edgeNodes.append(tempEdge)

        questionIdSet.append(edgeNode.questionNode(questionId, conceptId, timestamp, value_memory_matrix))
        self.student4question[studentId] = questionIdSet

        studentIdSet.append(edgeNode.studentNode(studentId))
        self.question4student[questionId] = studentIdSet

    def printStudentAnswerSeq(self, studentId) -> str:
        return "todo"

    def getSimilarStudentSeq(self) -> list:
        return list()