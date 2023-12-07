def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import asyncio
from openai import OpenAI, AsyncOpenAI
from nltk.translate.bleu_score import corpus_bleu
from functools import wraps
from bleurt import score
from numpy import mean, var, argmin


api_key = "Your API key"
data_folder = "data"
output_folder = "output"
batch_size = 1

translate_num = 1000
example_num = 0
dataset_name = 'zh-en'
#dataset_name = 'de-en'
#dataset_name = 'ja-en'
translator_model = 'gpt-3.5-turbo'
#translator_model = 'gpt-4'
#translator_model = 'facebook'
#translator_model = 'google'
surfix = ''
if batch_size > 1:
    surfix += '-batch-{}'.format(batch_size)
if example_num > 0:
    surfix += '-example-{}'.format(example_num)

limit_size = 200
if translator_model == 'gpt-4':
    limit_size = 20

just_evalueate = True
client = OpenAI(
    api_key=api_key,
)
async_client = AsyncOpenAI(
    api_key=api_key,
)
total_sentences = 0
translated_sentences = 0

def retry_on_timeout(retries=3, timeout=10):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return await asyncio.wait_for(func(*args, **kwargs), timeout)
                except asyncio.TimeoutError:
                    attempts += 1
                    if attempts == retries:
                        raise
                    print(f"Timeout occurred, retrying... (Attempt {attempts+1} of {retries})")
                    await asyncio.sleep(timeout)
        return wrapper
    return decorator


def call_chatgpt(question, model=translator_model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model=model,
    )
    answer = answer = chat_completion.choices[0].message.content
    return answer

@retry_on_timeout(retries=5, timeout=30)
async def async_call_chatgpt(question, model=translator_model):
    chat_completion = await async_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        model=model,
    )
    answer = answer = chat_completion.choices[0].message.content
    return answer


def gpt_translate(src_lines):
    question = "Translate following {} sentenses into English, just return the translate result and seperate by new line, make sure the number of lines returned is equal to number of input sentences:".format(len(src_lines))
    for src in src_lines:
        question += '\n' + src
    answer = call_chatgpt(question)
    return answer.split('\n')


async def async_gpt_translate(src_lines, examples=[]):
    example_prompt = ''
    if examples:
        example_prompt = ' I will give you some example for translation style reference,'
    if len(src_lines) > 1:
        question = "Translate following {} sentenses into English,{} each line is considered as one sentence, just return the translate result and seperate by new line:".format(len(src_lines),example_prompt)
    else:
        question = "Translate following sentense into English,{} only return the translate result:".format(example_prompt)
    if examples:
        question += '\nExamples:'
        for src_s, dst_s in examples:
            question += '\n{} ==> {}'.format(src_s, dst_s)
        question += '\nSentences:'
    for src in src_lines:
        question += '\n' + src
    
    answer = await async_call_chatgpt(question)
    dst_lines = answer.split('\n')
    dst_lines = [s for s in dst_lines if s.strip()]
    if len(dst_lines) != len(src_lines):
        print("Wrong dst lines length of {} vs {}, retry".format(
            len(dst_lines), len(src_lines)))
        for line in src_lines:
            print(line)
        for line in dst_lines:
            print(line)
        return await batch_translate(src_lines, max(len(src_lines)//2, 1), examples)
    global translated_sentences
    global total_sentences
    translated_sentences += len(dst_lines)
    print("{}/{} translated".format(translated_sentences, total_sentences),end = '\r')
    return dst_lines


def read_pairs(dataset_name):
    src_file = "{}/{}.src".format(data_folder, dataset_name)
    dst_file = "{}/{}.dst".format(data_folder, dataset_name)
    print("Reading data from {} and {}".format(src_file, dst_file))
    src_lines = open(src_file, encoding='utf-8').read().split('\n')
    dst_lines = open(dst_file, encoding='utf-8').read().split('\n')
    return list(zip(src_lines, dst_lines)), src_lines, dst_lines


def read_output(dataset_name):
    dst_file = "{}/{}-{}{}.dst".format(output_folder, dataset_name, translator_model, surfix)
    print("Reading output data from {}".format(dst_file))
    dst_lines = open(dst_file, encoding='utf-8').read().split('\n')
    return dst_lines


def save_output(dataset_name, lines):
    out_file = "{}/{}-{}{}.dst".format(output_folder, dataset_name, translator_model, surfix)
    with open(out_file, 'w') as fp:
        fp.write('\n'.join(lines))
    print("Output saved to "+out_file)


def evaluate(outputs, references):
    scorer = score.BleurtScorer()
    scores = scorer.score(references=references, candidates=outputs)
    worst_i = argmin(scores)
    print("Worst translate: \n{}\nvs\n{}".format(outputs[worst_i],references[worst_i]))
    print("BLEURT score:{}, var:{}, min:{}".format(mean(scores),var(scores),min(scores)))
    references = [[sent.split()] for sent in references]
    outputs = [sent.split() for sent in outputs]
    print("Calculating BLEU score...")
    bleu = corpus_bleu(references, outputs)
    print("BLEU score:{}".format(bleu))
    return bleu


async def batch_translate(src_lines, batch_size=batch_size, examples=[]):
    src_batchs = [src_lines[i:i + batch_size]
                  for i in range(0, len(src_lines), batch_size)]
    tasks = [async_gpt_translate(src_batch, examples) for src_batch in src_batchs]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

async def batch_translate_with_limit(src_lines, limit_size, batch_size=batch_size, examples=[]):
    limited_batchs = [src_lines[i:i + limit_size]
                  for i in range(0, len(src_lines), limit_size)]
    results = []
    for limited_batch in limited_batchs:
        results += await batch_translate(limited_batch, batch_size=batch_size, examples=examples)
    return results

async def main():
    pairs, src_lines, dst_lines = read_pairs(dataset_name)
    if not just_evalueate:
        src_lines, dst_lines = src_lines[:translate_num], dst_lines[:translate_num]
        global total_sentences
        total_sentences = len(src_lines)
        examples = pairs[1020:1020+example_num]
        gpt_dst_lines = await batch_translate_with_limit(src_lines,limit_size=limit_size, examples=examples)
        save_output(dataset_name, gpt_dst_lines)
        evaluate(gpt_dst_lines, dst_lines)
    else:
        gpt_dst_lines = read_output(dataset_name)
        dst_lines = dst_lines[:len(gpt_dst_lines)]
        evaluate(gpt_dst_lines, dst_lines)

asyncio.run(main())
