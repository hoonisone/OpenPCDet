from pathlib import Path

import yaml
from easydict import EasyDict

"""
    모델 학습에 필요한 설정(config)내용을 하나의 EasyDict 객체로 관리하는 파일
    속성을 편리하게 get, set하는 코드들이 있다.
"""

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """
    리스트의 설정 기능들을 config파일에 넣어준다.
    Args:
        cfg_list: cfg 객체에 저장할 속성들의 list
        config: config파일 객체(EasyDict)

    Returns:

    * cfg_list는 홀짝 단위로 key, value가 번갈아가며 나온다.
    * 반드시 짝수 길이를 가진다.
    * key는 config의 특정 속성을 a.b.c 형식으로 나타낸다.
    * value는 dict, list, singld 값을 가질 수 있는 것 같다.
    * key, value 둘 다 string을 디폴트 타입으로 갖는 것 같다.
    """

    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    # eval은 python의 built in 함수
    # eval(x:str)은 입력 스트링 x를 하나의 소스코드 처럼 인식하여 실행해주는 함수이다.
    # eval("print('hello')")를 하면 print('hello')가 실행된다.

    # literal_eval은 eval에 대해 파이썬 기본 자료형 연산만 허용하는 함수이다.
    # eval의 기능은 너무 위험하여 만들어진 기능이라 이해

    assert len(cfg_list) % 2 == 0 # 짝수 확인
        # 아마도 홀수 인자를 key, 짝수 인자를 value로 쓰려는 것 같다.

    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # list[a::b]는 list의 a 인덱스 부터 b만큼식 건너뛰어 sampling하겠다는 말
        # ex) [1, 2, 3, 4][1::2] = [2, 4]

        # ex) key = a.b.c     # dot(.)을 통해 특정 속성에 접근하는 표현
        # ex) vlaue = 10

        key_list = k.split('.')
        # dot(.)단위로 속성 경로를 분해

        d = config
        for subkey in key_list[:-1]:
            # 중간 경로(마지막 제외)에 대해

            assert subkey in d, 'NotFoundKey: %s' % subkey
            # 존재하는지 체크

            d = d[subkey]
            # 속성 들어가기
        # 위 for를 끝내면 d는 원하는 속성(실제 편집할 녀석)의 부모를 가리키고 있다.

        subkey = key_list[-1]
        # 편집될 대상(leaf)에 대한 key

        assert subkey in d, 'NotFoundKey: %s' % subkey
        # 존재 하는지 체크

        try:
            value = literal_eval(v)
            # json 형태의 string인 경우 literal_eval()를 통해 json객체로 만들 수 있다.
        except:
            value = v
            # 에러가 난다면 아마도 일반 const value(int 같은)인 경우~

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            # 조건 이해
            # (1) type(value) != type(d[subkey]) -> 새로운 값과 기존 값이 서로 다른 타입인가?

            # (2) isinstance(d[subkey], EasyDict) -> 기존 값이 EasyDict형태인가?
                # 이때 config 객체는 EasyDict이며 이는 config안에 어떤 dict객체도 EasyDict라는 말
                # 즉 EasyDict인지 체크하는 것은 dict인지 체크하는 것
                # 그 반대 경우는 int와 같은 단일 값

            # 즉 원래는 EasyDict이고 새로운 값은 EasyDict가 아닌 경우


            key_val_list = value.split(',')
            # value 에 (,)가 들어가나?... 왜?
            # 아마도 value가 dict인듯 그래서 ,로 먼저 구분한 뒤 :로 구분하는 듯

            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                # key:value 쌍을 :로 구분하여 key와 value로 나눈다.

                val_type = type(d[subkey][cur_key])
                # 기존 값의 type

                cur_val = val_type(cur_val)
                # int('1') 나 type(1)('1')나 동일하게 쓰이는듯 하다.
                # 새로운 값을 원래 값의 타입에 맞추겠다는 말

                d[subkey][cur_key] = cur_val
                # 값 수정

        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            # 새로운 값은 list가 아니고, 원래 값은 list인 경우

            val_list = value.split(',')
            # 요소 단위로 쪼갠다.

            for k, x in enumerate(val_list):
                # 각 요소에 대해

                val_list[k] = type(d[subkey][0])(x)
                # 들어갈 위치에 원래 값과 타입을 맞춘다.

            d[subkey] = val_list
            # type을 모두 맞춘 값들의 list를 통째로 변경
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            # 이외의 경우는 모두 새 값과 기존 값의 타입이 동일한 경우만 고려

            d[subkey] = value
            # 타입이 같기 때문에 한번에 교체

        # leaf 값이 dict, list, single인 경우 각각에 대해 따로 처리하는 코드이다.

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


cfg = EasyDict()
# EasyDict는 json객체를 [] 뿐만 아니라 dot(.)으로 접근 가능하게 해주며 재귀적으로 이용 가능하게 해준다.
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
