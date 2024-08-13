class ScoreTracker:
    def __init__(self, stop_after_epochs=5, initial_max=0):

        self.counter = 0
        self.max_score = initial_max
        self.stop_after_epochs = stop_after_epochs
        self.scores = []
        self.stop_training = False

    def __call__(self, score):
        
        if self.stop_after_epochs < 1:
            # deactivated -> always continue training
            return False
        
        self.max_score = max(self.scores) if len(self.scores) > 0 else self.max_score
        self.scores.append(score)
        
        if score <= self.max_score:
            self.counter += 1  # advance counter if max_score is not exceeded
        else: 
            self.counter = 0  # reset counter if max_score is exceeded

        if len(self.scores) >= self.stop_after_epochs:  
            #  after minimum number of epochs
            if self.counter >= self.stop_after_epochs:  
                # if max_score was not exceeded for threshold number of epochs
                self.stop_training = True
                
    def stop(self):
        return self.stop_training
    
    def print_summary(self, round_precision=3):
        last_score = round(self.scores[-1], round_precision)
        max_score = round(self.max_score, round_precision)
        score_diff = round(last_score - max_score, round_precision)
        
        print(f'last score: {last_score}')
        print(f'max score from previous epochs: {max_score}')
        if self.counter == 0:
            print('new max score achieved in this epoch')
        else: 
            print(f'max score achieved {self.counter} epochs ago')
        print(f'difference to previous max: {score_diff}')
        
        print(f'stop training: {self.stop_training}')