import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def run(model_name, question, context, sup_facts):
    # TODO remove tokens like [SEP]? tokens[0], tokens[12], tokens[len-1]
    # TODO kmeans clustering?
    tokenizer = BertTokenizer.from_pretrained(model_name)
    pretrained_weights = torch.load("./weights/babi.bin", map_location=torch.device('cpu'))

    model = BertForQuestionAnswering.from_pretrained(model_name,
                                                     state_dict=pretrained_weights, output_hidden_states=True,
                                                     return_dict=True)
    # model = BertForQuestionAnswering.from_pretrained(model_name, output_hidden_states=True, return_dict=True)
    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_index = input_ids.index(tokenizer.sep_token_id)

    sup_ids = tokenizer.encode(sup_facts)
    sup_tokens = tokenizer.convert_ids_to_tokens(sup_ids)
    # sup_start_id, sup_end_if = get_sup_ids(tokens, sup_tokens)

    # Segment A is the question, segment B is the answer?
    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0] * num_seg_a + [1] * num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # Output is a tuple with torch tensors
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    start_scores = output.start_logits
    end_scores = output.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = ' '.join(tokens[answer_start:answer_end + 1])
    print(answer)

    hidden_states = output.hidden_states

    for i, layer in enumerate(hidden_states):
        # Token embedding of one of the 118 tokens in the question text
        # Remove padding s
        hidden_layer = layer[0][:len(tokens)]
        hidden_layer = hidden_layer.squeeze(0)
        hidden_layer_np = hidden_layer.cpu().detach().numpy()

        reduction = PCA(n_components=2)
        reduced_layer = reduction.fit_transform(hidden_layer_np)
        question_tokens, context_tokens = get_tokens_lists(tokens)
        plot_hidden_states(reduced_layer, tokens, question_tokens, context_tokens, answer, i, sep_index)


# TODO Test
def get_sup_ids(tokens, sup_tokens):
    start_id = 0
    end_id = 0
    sequence = False
    index = 0
    for i in range(len(tokens)):
        if tokens[i] == sup_tokens[index]:
            if sequence:
                index += 1
            else:
                start_id = i
                sequence = True
        else:
            if sequence:
                end_id = i - 1
                sequence = False
    return start_id, end_id


def get_tokens_lists(tokens):
    question_tokens = []
    context_tokens = []
    for i in range(0, len(tokens)):
        if tokens[i] == '[SEP]':
            question_tokens = tokens[1:i]
            context_tokens = tokens[i + 1:len(tokens) - 1]
            break
    return question_tokens, context_tokens


def plot_hidden_states(reduced_layer, tokens, question_tokens, context_tokens, answer, layer_index, sep_index):
    # TODO Supporting facts tokens
    # TODO differentiate if token was in question or context (based on where it was -> string embedding first question then context)
    x_data = [x[0] for x in reduced_layer]
    y_data = [x[1] for x in reduced_layer]
    # iterate over all datapoint
    for i in range(len(x_data)):
        color = "gray"
        marker = "o"
        if tokens[i] in answer:
            color = "red"
            marker = "d"
        elif tokens[i] in question_tokens and i < 12:
            color = "orange"
            marker = "*"
        plt.scatter(x_data[i], y_data[i], c=color, marker=marker)
        plt.text(x_data[i] + 0.1, y_data[i] + 0.2, tokens[i], fontsize=6)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    filename = "layer" + str(layer_index) + ".pdf"
    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':
    squad_question = "What is a common punishment in the UK and Ireland?"

    squad_context = "Currently detention is one of the most common punishments in schools in the United States, the UK, Ireland, " \
              "Singapore and other countries. It requires the pupil to remain in school at a given time in the school day " \
              "(such as lunch, recess or after school); or even to attend school on a non-school day, e.g. Saturday detention " \
              "held at some schools. During detention,students normally have to sit in a classroom and do work, write lines or" \
              " a punishment essay, or sit quietly."

    squad_answer = "detention"
    # TODO save the indexes of sup facts in context
    supporting_facts = "Currently detention is one of the most common punishments in schools in the United States, the UK, Ireland, " \
                       "Singapore and other countries."

    babi_question = "What is Emily afraid of?"
    babi_context = "Wolves are afraid of cats.\nSheep are afraid of wolves.\nMice are afraid of sheep.\nGertrude is a mouse.\nJessica is a mouse.\nEmily is a wolf.\nCats are afraid of sheep.\nWinona is a wolf."
    babi_answer = "cats"

    hotpot_question = "What year did Guns N Roses perform a promo for a movie starring Arnold Schwarzenegger as a former New York Police detective?"
    hotpot_context = "Arnold Schwarzenegger filmography. Arnold Schwarzenegger is an actor who has appeared in over 30 films, and has also ventured into directing and producing. He began his acting career primarily with small roles in film and television. " \
                     "For his first film role, he was credited as \"Arnold Strong\", but was credited with his birth name thereafter. He has appeared mainly in action, and comedy films." \
                     " In addition to films and television, he has appeared in music videos for AC\/DC, Bon Jovi, and Guns N' Roses. Guns N' Roses discography." \
                     " Guns N' Roses is an American hard rock band formed in Los Angeles, California in 1985 by members of Hollywood Rose and L.A. Guns. The band has released six studio albums, two live albums, two compilation albums, two extended plays, seven video albums," \
                     " eighteen singles, twenty four music videos and one video single. Guns N' Roses signed a deal with Geffen Records in 1986, after the independently released EP \"Live ?!*@ Like a Suicide\" a year before. Its debut studio album \"Appetite for Destruction\" was " \
                     "released in 1987, reached the top of the \"Billboard\" 200 and sold 18 million units in the United States and approximately 33 million units worldwide. Hammerjack's. Hammerjacks Concert Hall and Nightclub was a large concert hall in downtown Baltimore through the 1980s and into " \
                     "the 1990s owned by Louis J. Principio III The club attracted many big-name national acts, but also showcased many rising stars in the music world.The bands ranged from punk, glam, and heavy metal acts most commonly associated with the venue (e.g., Guns n Roses, Kix, Ratt, Skid Row or Extreme) " \
                     "to pop (e.g., Badfinger) and alternative rock groups (e.g., Goo Goo Dolls). The club was often frequented by hard core patrons and musicians donning big hair, leather, lace, spandex, and heavy makeup, and was considered a \"hard rock shrine.\" Hamerjacks, however, attracted audiences with other attire as well." \
                     " It was torn down on June 12, 1997 to make way for M&T Bank Stadium parking lot. Hammerjacks was billed as \"The largest nightclub on the east coast.\"Steven Adler. Steven Adler (born Michael Coletti; January 22, 1965) is an American musician. He is best known as the former drummer and co-songwriter of the hard rock " \
                     "band Guns N' Roses, with whom he achieved worldwide success in the late 1980s. Adler was fired from Guns N' Roses over his heroin addiction in 1990, following which he reformed his old band Road Crew and briefly joined BulletBoys, which both proved unsuccessful. During the 2000s, Adler was the drummer of the band Adler's " \
                     "Appetite, and from 2012, he had held the same position in the band Adler. In early 2017, Steven Adler declared that he has no intention to continue with the band, and that the band has now dissolved, and the reason is his lack of interest in performing during poorly attended concerts.He appeared on the second and fifth seasons " \
                     "of the reality TV show \"Celebrity Rehab with Dr. Drew\", as well as on the first season of its spin-off \"Sober House\". He was inducted into the Rock and Roll Hall of Fame in 2012 as a member of Guns N' Roses.Faction Punk. Faction With Jason Ellis is an uncensored hard rock, punk, hip hop, and heavy metal music mixed channel on Siruis" \
                     " XM Satellite Radio. Until mid-July 2017, Faction appeared on Sirius XM channel 41. In mid-July 2017, Faction was temporarily replaced by Guns N Roses radio. After August 16, 2017, channel 41 was rebranded to Turbo, Sirius XM's channel for hard rock from the 1990s and 2000s. Faction moved to channel 314, Turbo's previous channel. Faction" \
                     " is currently available on select Sirius XM radios, Sirius XM streaming, and the Sirius XM smartphone app. True Lies. True Lies is a 1994 American action film written, co-produced and directed by James Cameron, starring Arnold Schwarzenegger, Jamie Lee Curtis, Tom Arnold, Art Malik, Tia Carrere, Bill Paxton, Eliza Dushku, Grant Heslov and " \
                     "Charlton Heston. It is a loose remake of the 1991 French comedy film \"La Totale!\" The film follows U.S. government agent Harry Tasker (Schwarzenegger), who balances his life as a spy with his familial duties. Last Action Hero. Last Action Hero is a 1993 American fantasy action comedy film directed and produced by John McTiernan.It is a satire of the action genre" \
                     " and associated clich\u00e9s, containing several parodies of action films in the form of films within the film.The film stars Arnold Schwarzenegger as Jack Slater, a Los Angeles police detective within the \"Jack Slater\" action film franchise. Austin O'Brien co-stars as a boy magically transported into the \"Slater\" universe. Schwarzenegger also served as the film's" \
                     " executive producer and plays himself as the actor portraying Jack Slater, and Charles Dance plays an assassin who escapes from the \"Slater\" world into the real world.End of Days (film). End of Days is a 1999 American fantasy action horror thriller film directed by Peter Hyams and starring Arnold Schwarzenegger, Gabriel Byrne, Robin Tunney, Kevin Pollak, Rod Steiger, CCH " \
                     "Pounder, and Udo Kier. The film follows former New York Police Department detective Jericho Cane (Schwarzenegger) after he saves a banker (Byrne) from an assassin, finds himself embroiled in a religious conflict, and must protect an innocent young woman (Tunney) who is chosen by evil forces to conceive the Antichrist with Satan.Oh My God (Guns N' Roses song). \"Oh My God\" is a " \
                     "song by Guns N' Roses released in 1999 on the soundtrack to the film \"End of Days\". The song was sent out to radio stations in November 1999 as a promo for the soundtrack and the band. Despite being the band's first recorded release in almost five years, it was never issued as a stand-alone single for public retail. Get Christie Love!. Get Christie Love! is a 1974 made-for-television film and" \
                     " subsequent crime drama TV series starring Teresa Graves as an undercover female police detective who is determined to overthrow a drug ring.This film is based on Dorothy Uhnak's crime-thriller novel \"The Ledger\". However, the main character \"Christie Opara\"\u2014a white, New York Police detective\u2014was dropped completely and \"Christie Love\" emerged."
    hotpot_answer = "1999"

    # model_name = 'bert-large-uncased-whole-word-masking-finetuned-weights'
    model_name = 'csarron/bert-base-uncased-squad-v1'
    # model_name = "bert-base-uncased"
    run(model_name, babi_question, babi_context, supporting_facts)
