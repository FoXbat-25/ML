import sys

def error_msg_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_msg = f"Error in {file_name} line number {exc_tb.tb_lineno}; error {str(error)}"
    return error_msg

class customException(Exception):
    def __init__(self, error_msg, error_detail:sys):
        super().__init__(error_msg)
        self.error_msg = error_msg_detail(error_msg, error_detail=error_detail)

    def __str__(self):
        return self.error_msg
    

# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         raise customException(e, sys)