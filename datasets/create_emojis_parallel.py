import urllib.request
import urllib.error
import numpy as np
import numpy.random as random
import multiprocessing
import os
import props as pr

DEBUG = False

num_emojis = 1000000

MALE = 1
FEMALE = 2

def create_emoji(i):
    # choose gender 50/50
    gender = MALE if random.random() < 0.5 else FEMALE
    if DEBUG: print('gender=' + str(gender))

    #
    # choose non pd2 properties
    #
    # face shape (uniform)
    proportion_ind = random.randint(0, high=len(pr.proportion))
    proportion = pr.proportion[proportion_ind]
    if DEBUG: print('proprtion=' + str(proportion))
    # build colour string
    colours_string = '{'
    # choose skin color (uniform)
    skin_color_arr = pr.skin_color[random.randint(0, high=len(pr.skin_color))]
    colours_string += '\"ffcc99\":' + str(skin_color_arr[0]) + ',\"ff9866\":' + str(skin_color_arr[1])
    if DEBUG: print('colours=' + colours_string)
    # choose hair/eyebrow/beard color (they will be the same, uniform)
    hair_color_ind = random.randint(0, high=len(pr.hair_color))
    hair_color_arr = pr.hair_color[hair_color_ind]
    colours_string += ',\"926715\":' + str(hair_color_arr[0]) + ',\"6f4b4b\":' + str(hair_color_arr[1]) + ',\"4f453e\":' + str(hair_color_arr[1])
    if DEBUG: print('colours=' + colours_string)
    if gender == FEMALE:
        # choose if has lipticks (50/50)
        has_lipstick = True if random.random() < 0.5 else False
        if has_lipstick:
            # choose lipticks color (uniform)
            lipstick_color = pr.lipstick_color[random.randint(0, high=len(pr.lipstick_color))]
            colours_string += ',\"ff9866\":' + str(lipstick_color)
            if DEBUG: print('colours=' + colours_string)

    # choose eye color (uniform)
    eye_color = pr.eye_color[random.randint(0, high=len(pr.eye_color))]
    colours_string += ',\"36a7e9\":' + str(eye_color)

    colours_string += '}'
    if DEBUG: print('colours=' + colours_string)

    #
    # build pd2 string
    #
    pd2_string = '{'
    # things that are same for both genders
    # eyebrows
    eyebrows = pr.eyebrows[random.randint(0, high=len(pr.eyebrows))]
    pd2_string += '\"brow_L\":\"brow_n{}\",\"brow_R\":\"brow_n{}\"'.format(eyebrows, eyebrows)
    if DEBUG: print('pd2=' + pd2_string)
    # eyes
    eyes = pr.eyes[random.randint(0, high=len(pr.eyes))]
    pd2_string += ',\"eye_L\":\"eye_n{}\",\"eye_R\":\"eye_n{}\",\"eyelines_L\":\"eye_n{}\",\"eyelines_R\":\"eye_n{}\",\"eyelid_L\":\"eyelid_n1_{}\",\"eyelid_R\":\"eyelid_n1_{}\"'.format(eyes, eyes, eyes, eyes, eyes, eyes)
    if DEBUG: print('pd2=' + pd2_string)
    # nose
    nose = pr.noses[random.randint(0, high=len(pr.noses))]
    pd2_string += ',\"nose\":\"nose_n{}\"'.format(nose)
    if DEBUG: print('pd2=' + pd2_string)
    # mouth
    mouth_arr = pr.mouths[random.randint(0, high=len(pr.mouths))]
    pd2_string += ',\"mouth\":\"mouth_n{}\",\"tongue\":\"tongue_n1_{}\"'.format(mouth_arr[0], mouth_arr[1])
    if DEBUG: print('pd2=' + pd2_string)
    # ears
    ears = pr.ears[random.randint(0, high=len(pr.ears))]
    pd2_string += ',\"ear_L\":\"ear_n{}\",\"ear_R\":\"ear_n{}\"'.format(ears, ears)
    if DEBUG: print('pd2=' + pd2_string)
    # glasses (20% chance)
    has_glasses = True if random.random() < 0.2 else False
    if has_glasses:
        glasses = pr.glasses[random.randint(0, high=len(pr.glasses))]
        pd2_string += ',\"glasses\":\"glasses_n{}\"'.format(glasses)
        if DEBUG: print('pd2=' + pd2_string)
    # now gender specific attributes
    if gender == MALE:
        # jaw
        jaw = pr.jaw_male[random.randint(0, high=len(pr.jaw_male))]
        pd2_string += ',\"jaw\":\"jaw_n{}\"'.format(jaw)
        if DEBUG: print('pd2=' + pd2_string)
        # choose if has beard (30% chance)
        has_beard = True if random.random() < 0.3 else False
        if has_beard:
            facial_hair_arr = pr.facial_hair[random.randint(0, high=len(pr.facial_hair))]
            if facial_hair_arr[1] == 0:
                # stachin is blank
                pd2_string += ',\"beard\":\"beard_n{}_1\",\"stachin\":\"stachin_blank\",\"stachout\":\"stachout_n{}_1\"'.format(facial_hair_arr[0], facial_hair_arr[2])
            else:
                # regular
                pd2_string += ',\"beard\":\"beard_n{}_1\",\"stachin\":\"stachin_n{}_1\",\"stachout\":\"stachout_n{}_1\"'.format(facial_hair_arr[0], facial_hair_arr[1], facial_hair_arr[2])
            if DEBUG: print('pd2=' + pd2_string)
        # hair (40% short, 30% med, 17.5% long, 10% bald, 2.5% wacky)
        hair_sample = random.random()
        if hair_sample < 0.4:
            hair = pr.male_hair_short[random.randint(0, high=len(pr.male_hair_short))]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_blank\"'.format(hair, hair, hair)
        elif hair_sample < 0.7:
            hair_ind = random.randint(0, high=len(pr.male_hair_med))
            hair = pr.male_hair_med[hair_ind]
            has_bottom = pr.male_hair_med_bottom[hair_ind]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_{}\"'.format(hair, hair, hair, hair if has_bottom else 'blank')
        elif hair_sample < 0.875:
            hair_ind = random.randint(0, high=len(pr.male_hair_long))
            hair = pr.male_hair_long[hair_ind]
            has_bottom = pr.male_hair_long_bottom[hair_ind]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_{}\"'.format(hair, hair, hair, hair if has_bottom else 'blank')
        elif hair_sample < 0.975:
            hair_ind = random.randint(0, high=len(pr.male_hair_bald))
            hair = pr.male_hair_bald[hair_ind]
            is_standard = pr.male_hair_bald_standard[hair_ind]
            front_blank = pr.male_hair_bald_frontblank[hair_ind]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_{}\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_blank\"'.format(hair, 'standard' if is_standard else hair, hair, 'blank' if front_blank else hair)
        else:
            hair = pr.hair_wacky[random.randint(0, high=len(pr.hair_wacky))]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_blank\"'.format(hair, hair, hair)
        if DEBUG: print('pd2=' + pd2_string)

    if gender == FEMALE:
        # jaw
        jaw = pr.jaw_female[random.randint(0, high=len(pr.jaw_female))]
        pd2_string += ',\"jaw\":\"jaw_n{}\"'.format(jaw)
        if DEBUG: print('pd2=' + pd2_string)
        # eyelashes
        eyelashes = pr.eyelashes[random.randint(0, high=len(pr.eyelashes))]
        if eyelashes == 0:
            pd2_string += ',\"eyelash_L\":\"_blank\",\"eyelash_R\":\"_blank\"'
        else:
            pd2_string += ',\"eyelash_L\":\"eyelash_n{}_1\",\"eyelash_R\":\"eyelash_n{}_1\"'.format(eyelashes, eyelashes)
        if DEBUG: print('pd2=' + pd2_string)
        # hair (17.5% short, 30% med, 50% long, 2.5% wacky)
        hair_sample = random.random()
        if hair_sample < 0.175:
            hair_ind = random.randint(0, high=len(pr.female_hair_short))
            hair = pr.female_hair_short[hair_ind]
            has_bottom = pr.female_hair_short_bottom[hair_ind]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_{}\"'.format(hair, hair, hair, hair if has_bottom else 'blank')
        elif hair_sample < 0.475:
            hair_ind = random.randint(0, high=len(pr.female_hair_med))
            hair = pr.female_hair_med[hair_ind]
            has_bottom = pr.female_hair_med_bottom[hair_ind]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_{}\"'.format(hair, hair, hair, hair if has_bottom else 'blank')
        elif hair_sample < 0.975:
            hair_ind = random.randint(0, high=len(pr.female_hair_long))
            hair = pr.female_hair_long[hair_ind]
            is_blank = pr.female_hair_long_blank[hair_ind]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_{}\"'.format(hair, hair, hair, 'blank' if is_blank else hair)
        else:
            hair = pr.hair_wacky[random.randint(0, high=len(pr.hair_wacky))]
            pd2_string += ',\"cranium\":\"cranium_{}\",\"forehead\":"forehead_standard\",\"hair_back\":\"hair_back_{}\",\"hair_front\":\"hair_front_{}\",\"hairbottom\":\"hairbottom_blank\"'.format(hair, hair, hair)
        if DEBUG: print('pd2=' + pd2_string)

    pd2_string += '}'
    if DEBUG: print('pd2=' + pd2_string)

    # put together entire string with correct user id based on gender
    user_id = '371434407_1_s1' if gender == MALE else '122369401_1_s1'
    request = 'https://render.bitstrips.com/render/6688424/' + user_id + '-v1.png?colours=' + colours_string + '&pd2=' + pd2_string + '&head_rotation=0&proportion='+ str(proportion) +'&sex=' + str(gender) + '&scale=0.382&style=1'
    if DEBUG: print(request)

    # call url and save image
    try:
        if os.path.exists('./emoji_data/emoji_{}.png'.format(i)):
            print("skip" + './emoji_data/emoji_{}.png'.format(i))
            return
        urllib.request.urlretrieve(request, './emoji_data/emoji_{}.png'.format(i))
        i += 1
    except urllib.error.HTTPError:
        print('Error for request: ' + request)

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
    pool.map(create_emoji, range(0, num_emojis))
    pool.close()
    pool.join()
