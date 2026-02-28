
import regex as re
import re


MAX_ANSWER_LEN = 100
def clean_and_repair_response(response: str, ground_truth: str) -> tuple[str, str]:
    """
    解析、清洗、修复和校验模型的 Response。
    现在使用强大的 grade() 进行数学等价性校验，而不是死板的字符串比对。
    """
    # 1. 基础标签数量校验
    tags = ["<think>", "</think>", "<answer>", "</answer>"]
    for tag in tags:
        if response.count(tag) != 1:
            return None, f"tag_count_error_{tag.strip('<> /')}"

    # 2. 定位标签位置
    t_start = response.find("<think>")
    t_end = response.find("</think>")
    a_start = response.find("<answer>")
    a_end = response.find("</answer>")

    if not (t_start < t_end < a_start < a_end):
        return None, "wrong_tag_order"

    think_content = response[t_start + 7: t_end]
    answer_content = response[a_start + 8: a_end]

    if not think_content.strip() or not answer_content.strip():
        return None, "empty_content"

    # 3. 防作弊检查 (思考过程不准提前框定答案)
    if "\\boxed" in think_content:
        return None, "boxed_in_think"

    # 4. 强制要求回答区必须有 \boxed (对齐你的 grader 逻辑)
    if "\\boxed" not in answer_content:
        return None, "no_boxed_in_answer"


    # 6. 尽力修复机制 (瘦身与排版)
    is_fixed = False

    # 如果通过了准确性校验，但内容太长（比如带了解释文本），进行提取瘦身
    if len(answer_content) > MAX_ANSWER_LEN:
        boxed_match = re.search(r"(\\boxed\{.*?\})", answer_content)
        if boxed_match:
            answer_content = boxed_match.group(1)
            is_fixed = True

    # 如果答案区有换行符，清除非法换行
    if "\n" in answer_content.strip():
        answer_content = answer_content.replace("\n", " ").strip()
        is_fixed = True

    # 7. 重组返回
    if is_fixed:
        response = f"<think>\n{think_content.strip()}\n</think>\n<answer>{answer_content}</answer>"
        return response, "fixed_and_passed"

    return response, "clean_passed"
