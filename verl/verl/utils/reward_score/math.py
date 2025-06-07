# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py


import re


def extract_after_assistant(response_str):
    """
    提取 Assistant: 或 <|im_start|>assistant 之后的内容。
    优先查找 Assistant:，如果找不到再查找 <|im_start|>assistant。
    """
    if not response_str:
        return None
    
    marker_1 = "Assistant:"
    idx_1 = response_str.find(marker_1)
    if idx_1 != -1:
        return response_str[idx_1 + len(marker_1):].strip()
    
    marker_2 = "<|im_start|>assistant"
    idx_2 = response_str.find(marker_2)
    if idx_2 != -1:
        return response_str[idx_2 + len(marker_2):].strip()
    
    return None


def extract_answer_block(response_str):
    """
    提取 <answer> 标签内的内容，并去除首尾空白字符。
    """
    if not response_str:
        return None
    match = re.search(r"<answer>(.*?)</answer>", response_str, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return content if content else None
    return None


def valid_format(response_str):
    """
    检查输出格式是否符合要求：
    1. 必须且仅出现一次 <think>、</think>、<answer>、</answer> 四个标签，
       且顺序严格为：<think>、</think>、<answer>、</answer>。
    2. <think></think> 与 <answer></answer> 标签中的内容不能为空（剔除空白字符后）。
    """
    if not response_str:
        return False

    think_start = response_str.find("<think>")
    think_end = response_str.find("</think>")
    answer_start = response_str.find("<answer>")
    answer_end = response_str.find("</answer>")

    if (think_start == -1 or think_end == -1 or answer_start == -1 or answer_end == -1 or
        response_str.count("<think>") != 1 or response_str.count("</think>") != 1 or
        response_str.count("<answer>") != 1 or response_str.count("</answer>") != 1):
        return False

    if not (think_start < think_end < answer_start < answer_end):
        return False

    think_content = response_str[think_start + len("<think>"):think_end].strip()
    answer_content = response_str[answer_start + len("<answer>"):answer_end].strip()

    if not think_content or not answer_content:
        return False

    return True


def compute_score(solution_str, ground_truth) -> float:
    """
    奖励函数：
      格式奖励 (Sformat):
         +1  if format is correct;
         -1  if format is incorrect.
      答案奖励 (Sanswer):
         +2    if the answer fully matches ground_truth;
         -1.5  if the answer partially mismatches ground_truth;
         -2    if the answer cannot be parsed or is missing.
    """
    response_str = extract_after_assistant(solution_str)
    Sformat = 1 if response_str and valid_format(response_str) else -1

    try:
        answer_block = extract_answer_block(response_str)
        if answer_block is None:
            raise ValueError("无法提取 <answer> 标签内的内容")
        
        boxed_answer = last_boxed_only_string(answer_block)
        if boxed_answer is None:
            raise ValueError("无法提取 boxed 答案")
        
        final_answer = remove_boxed(boxed_answer)
        Sanswer = 2 if is_equiv(final_answer, ground_truth) else -1.5
    except Exception:
        Sanswer = -2

    return Sformat + Sanswer


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
