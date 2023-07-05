import streamlit as st
from audiorecorder import audiorecorder
import whisper
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re



# Define the labeled examples
examples = [
   ("We are actively seeking talented individuals to join our growing startup.", "Talent"),
("Our startup is looking for exceptional talent with a strong background in software engineering.", "Talent"),
("We value creativity and innovation, and we are eager to attract talented individuals who can contribute to our team.", "Talent"),
("As a startup, we prioritize hiring top talent who can adapt quickly to a fast-paced environment.", "Talent"),
("Our company believes in nurturing talent and providing opportunities for growth within the organization.", "Talent"),
("We are seeking talented individuals who possess a passion for problem-solving and can bring fresh perspectives to our startup.", "Talent"),
("If you have a track record of success and are eager to work in a dynamic startup environment, we want to hear from you.", "Talent"),
("Join our team of talented professionals who are dedicated to pushing the boundaries of technology in the startup landscape.", "Talent"),
("We have high standards when it comes to talent acquisition and look for candidates who demonstrate exceptional skills and potential.", "Talent"),
("At our startup, we believe that diversity and inclusion are crucial in attracting top talent and fostering a thriving work culture.", "Talent"),
("We are actively seeking talented individuals to join our growing startup.", "Talent"),
("Our startup is expanding, and we're on the lookout for talented individuals who can help drive our growth and success.", "Talent"),
("If you're a talented go-getter with a passion for innovation, we want you on our team.", "Talent"),
("We're actively recruiting top-tier talent to join our dynamic startup and be part of our journey towards success.", "Talent"),
("Calling all talented individuals! Join our rapidly growing startup and make your mark in the industry.", "Talent"),
("We believe that talented people are the backbone of any successful startup. Join us and be part of our success story.", "Talent"),
("If you're a talented problem-solver with a hunger for challenges, we have an exciting role waiting for you in our startup.", "Talent"),
("We're building a team of talented individuals who share our vision and are ready to make a significant impact in the market.", "Talent"),
("Do you have what it takes to be a part of a fast-paced startup environment? We're seeking talented individuals like you.", "Talent"),
("Our startup is on a mission to revolutionize the industry, and we need talented professionals to help us achieve our goals.", "Talent"),
("If you're a talented self-starter who thrives in a collaborative environment, we want to hear from you.", "Talent"),
("We're looking for talented individuals who are passionate about our product and eager to contribute to its success.", "Talent"),
("Join our team of talented innovators and work on cutting-edge projects that will shape the future of our startup.", "Talent"),
("Talented minds wanted! Be part of our startup's success by bringing your unique skills and ideas to the table.", "Talent"),
("Are you a talented leader ready to take on new challenges? Join us as we grow and build a remarkable startup together.", "Talent"),
("We're actively seeking talented individuals who can bring fresh perspectives and help us stay ahead in a competitive market.", "Talent"),
("If you're a talented communicator with a knack for building relationships, join our startup and make an impact in our industry.", "Talent"),
("At our startup, we value talent and offer a supportive environment where you can thrive and reach your full potential.", "Talent"),
("Join our startup and work alongside a team of talented professionals who are passionate about making a difference.", "Talent"),
("We're looking for talented individuals who are eager to learn, grow, and contribute to the success of our startup.", "Talent"),
("If you have a talent for problem-solving and a drive for innovation, we want you to join our dynamic startup.", "Talent"),
("Join our startup and be part of a team that values talent, creativity, and collaboration.", "Talent"),
("We're actively searching for talented individuals who can contribute their skills and expertise to fuel our startup's success.", "Talent"),
("Are you a talented professional seeking a rewarding career in a fast-paced startup environment? Look no further!", "Talent"),
("Our startup is dedicated to attracting top talent and creating a culture of excellence and growth.", "Talent"),
("If you're a talented individual with a passion for entrepreneurship, our startup provides the perfect platform for your success.", "Talent"),
("We believe that talent knows no boundaries, and we welcome individuals from all backgrounds to join our startup.", "Talent"),
("Our startup is on the lookout for talented problem-solvers who can think outside the box and drive innovation.", "Talent"),
("Join our team of talented visionaries and shape the future of our startup as we disrupt the industry.", "Talent"),
("We're seeking talented individuals who are not only skilled in their respective fields but also share our startup's values and vision.", "Talent"),
("At our startup, we foster a culture of continuous learning and development to help our talented employees reach their full potential.", "Talent"),
("If you're a talented leader with a track record of success, join our startup and help us build a thriving organization.", "Talent"),
("We recognize that attracting and retaining top talent is key to our startup's growth and are committed to creating an environment where talent can flourish.", "Talent"),
("Join our startup and become part of a community where your talents are celebrated and your ideas are valued.", "Talent"),
("Are you a talented multitasker with a passion for taking on new challenges? Join our startup and make a real impact.", "Talent"),
("We're seeking talented individuals who are not afraid to take risks, challenge the status quo, and drive our startup to new heights.", "Talent"),
("Join our startup and work alongside a diverse team of talented individuals who inspire and support each other.", "Talent"),
("We believe in empowering our talented employees by providing them with the tools, resources, and opportunities they need to succeed.", "Talent"),
("Our startup is committed to creating a work environment that encourages collaboration, creativity, and the nurturing of talent.", "Talent"),
("If you're a talented professional looking for a dynamic startup where you can make a meaningful impact, we would love to hear from you.", "Talent"),
("We're actively searching for talented individuals who are passionate about our startup's mission and ready to make a difference.", "Talent"),
("Join our startup and be part of a team where your talent is valued, and your contributions have a direct impact on our success.", "Talent"),
("Are you a talented problem-solver with a keen eye for detail? Our startup is looking for individuals like you to join our ranks.", "Talent"),
("We're building a culture that celebrates diversity and fosters the growth of talented individuals from all backgrounds.", "Talent"),
("If you're a talented communicator with the ability to inspire and lead, our startup offers an exciting leadership opportunity.", "Talent"),
("Join our startup and work on cutting-edge projects alongside a team of talented professionals who share your passion for innovation.", "Talent"),
("We're seeking talented individuals who can bring fresh ideas and perspectives to our startup and contribute to our ongoing success.", "Talent"),
("Our startup believes in investing in talent development, providing continuous learning opportunities for our dedicated employees.", "Talent"),
("Are you a talented team player who thrives in a collaborative environment? Join our startup and make an impact through collective success.", "Talent"),
("We're actively recruiting talented individuals who have a growth mindset and are eager to take on new challenges in our startup.", "Talent"),
("Join our startup and be part of a culture that celebrates and rewards the entrepreneurial spirit and the talents it brings.", "Talent"),
("We believe that our startup's success lies in attracting and retaining top talent, and we're committed to creating an environment where talent can thrive.", "Talent"),
("If you're a talented visionary with a knack for spotting trends, our startup provides the perfect platform for you to shape the future.", "Talent"),
("We're looking for talented individuals who are not afraid to push boundaries, experiment, and create new possibilities in our startup.", "Talent"),
("Join our team of talented problem-solvers and work on innovative solutions that address real-world challenges in our industry.", "Talent"),
("At our startup, we believe that every individual has unique talents to offer, and we're dedicated to unlocking their full potential.", "Talent"),
("We're seeking talented professionals who are passionate about delivering exceptional user experiences and driving customer satisfaction.", "Talent"),
("Join our startup and embark on a journey of personal and professional growth alongside a supportive community of talented individuals.", "Talent"),
("Are you a talented data analyst with a passion for uncovering insights? Our startup is looking for individuals like you to join our analytics team.", "Talent"),
("We're on the lookout for talented individuals who thrive in a fast-paced environment and are excited to contribute to our startup's rapid growth.", "Talent"),
("Invest in our startup and become part of a groundbreaking venture poised for exceptional growth.", "Investment"),
("We're seeking strategic investors who share our vision and want to be part of the next big success story.", "Investment"),
("Join our journey by investing in our startup and capitalize on the potential for substantial returns.", "Investment"),
("Invest in our innovative startup and help us disrupt the industry with our groundbreaking solutions.", "Investment"),
("We're actively seeking investment partners who believe in our mission and want to be part of our exciting growth trajectory.", "Investment"),
("Invest in our startup and be part of a lucrative opportunity that has the potential to reshape the market.", "Investment"),
("Join our network of investors and gain access to a portfolio of high-potential startups with exceptional growth prospects.", "Investment"),
("We're inviting investors to join us on our mission to revolutionize the industry and generate significant returns on investment.", "Investment"),
("Invest in our startup and be at the forefront of innovation, driving positive change and reaping the rewards.", "Investment"),
("We're looking for visionary investors who understand the potential of our startup and want to make a strategic investment.", "Investment"),
("Join our investor community and gain exclusive access to exciting investment opportunities in the fast-growing startup ecosystem.", "Investment"),
("Invest in our startup and leverage the expertise of our seasoned team to maximize your return on investment.", "Investment"),
("We're seeking strategic investors who can provide more than just capital - investors who can contribute valuable industry insights and networks.", "Investment"),
("Join us in fueling the growth of our startup by making an impactful investment that will propel us to new heights.", "Investment"),
("Invest in our startup and play a vital role in shaping the future of our industry through innovation and disruptive solutions.", "Investment"),
("We're seeking like-minded investors who believe in the power of technology and want to be part of our startup's success story.", "Investment"),
("Join our network of visionary investors and gain access to a pipeline of promising startups with high growth potential.", "Investment"),
("Invest in our startup and diversify your portfolio with a high-potential investment opportunity in a rapidly evolving market.", "Investment"),
("We're inviting investors to join us in building a sustainable future through impactful investments in our socially responsible startup.", "Investment"),
("Invest in our startup and enjoy the excitement of being part of an entrepreneurial journey with limitless possibilities for success.", "Investment"),
("Investments play a crucial role in fueling the growth and expansion of startups, providing the necessary capital to fund operations, development, and scaling.", "Investment"),
("Startups rely on investments to attract top talent, build a skilled team, and foster innovation through research and development.", "Investment"),
("Investments provide startups with the financial resources needed to penetrate the market, establish their brand, and gain a competitive edge.", "Investment"),
("With the help of investments, startups can invest in marketing and customer acquisition strategies to reach a wider audience and drive business growth.", "Investment"),
("Investments enable startups to invest in technology infrastructure, equipment, and tools, empowering them to operate efficiently and deliver high-quality products or services.", "Investment"),
("Startups can leverage investments to expand their product or service offerings, diversify revenue streams, and adapt to evolving market trends.", "Investment"),
("Investments provide startups with the opportunity to attract experienced advisors and mentors who can provide valuable guidance and industry expertise.", "Investment"),
("With the support of investments, startups can conduct market research, gather insights, and make data-driven decisions to optimize their operations and offerings.", "Investment"),
("Investments allow startups to forge strategic partnerships, collaborations, and acquisitions, facilitating access to new markets, customers, and distribution channels.", "Investment"),
("Startups can use investments to build a robust intellectual property portfolio, protect their innovations, and gain a competitive advantage in the market.", "Investment"),
("Investments provide startups with the financial stability and runway needed to weather challenges, overcome obstacles, and sustain long-term growth.", "Investment"),
("By securing investments, startups can attract media attention, enhance their brand visibility, and establish credibility within their industry.", "Investment"),
("Investments can provide startups with the resources to expand their physical infrastructure, such as office spaces or manufacturing facilities, to support their growth.", "Investment"),
("Startups can leverage investments to implement scalable business models, streamline operations, and optimize processes for efficiency and profitability.", "Investment"),
("Investments allow startups to attract partnerships and collaborations with established companies, opening doors to shared resources, expertise, and market access.", "Investment"),
("With the help of investments, startups can embark on research and development initiatives, driving innovation and creating groundbreaking solutions.", "Investment"),
("Investments enable startups to hire specialized talent, build cross-functional teams, and foster a culture of creativity and entrepreneurship.", "Investment"),
("Startups can use investments to develop and execute marketing strategies, raise brand awareness, and engage with their target audience.", "Investment"),
("Investments provide startups with the flexibility to iterate and pivot their business models based on market feedback and changing customer needs.", "Investment"),
("By attracting investments, startups gain the confidence of potential customers, partners, and stakeholders, enhancing their reputation and market position.", "Investment"),
("Investing in startups provides an opportunity to participate in the early stages of groundbreaking ideas and potentially achieve significant financial returns.", "Investment"),
("Startups rely on investments to accelerate product development, launch new features, and stay ahead of competitors in a rapidly evolving market.", "Investment"),
("Investments fuel the research and development efforts of startups, enabling them to innovate and bring disruptive solutions to the market.", "Investment"),
("By investing in startups, you become a key player in shaping the future and supporting the growth of the entrepreneurial ecosystem.", "Investment"),
("Startups use investments to build a solid infrastructure, implement scalable systems, and establish a strong foundation for sustainable growth.", "Investment"),
("Investments provide startups with the necessary runway to focus on customer acquisition, market penetration, and revenue generation.", "Investment"),
("Investing in startups is an opportunity to support visionary founders, passionate teams, and their mission to solve real-world problems.", "Investment"),
("Startups rely on investments to attract strategic partners, expand distribution networks, and reach new markets globally.", "Investment"),
("Investments enable startups to navigate challenges, seize market opportunities, and adapt their business strategies for long-term success.", "Investment"),
("Investing in startups allows you to be part of the entrepreneurial journey, experiencing the highs and lows and celebrating milestones together.", "Investment"),
("Startups utilize investments to enhance their product or service offerings, improve user experience, and deliver exceptional value to customers.", "Investment"),
("Investments fuel the hiring efforts of startups, empowering them to recruit top talent, build diverse teams, and foster a culture of innovation.", "Investment"),
("Investing in startups offers the chance to support disruptive technologies that have the potential to reshape industries and improve lives.", "Investment"),
("Startups use investments to implement effective marketing strategies, create brand awareness, and establish a strong market presence.", "Investment"),
("Investments provide startups with the necessary capital to scale their operations, expand into new geographies, and capture a larger market share.", "Investment"),
("Investing in startups allows you to diversify your investment portfolio, tapping into high-growth opportunities beyond traditional asset classes.", "Investment"),
("Startups leverage investments to build robust customer acquisition channels, drive user engagement, and foster customer loyalty.", "Investment"),
("Investments enable startups to invest in data analytics, AI, and machine learning capabilities, gaining valuable insights and competitive advantages.", "Investment"),
("Investing in startups is a way to support local economies, job creation, and foster innovation-driven ecosystems.", "Investment"),
("Startups utilize investments to strengthen their intellectual property portfolio, protecting their innovations and establishing a defensible market position.", "Investment"),
("Investments provide startups with the financial flexibility to experiment, iterate, and pivot their business models based on market feedback.", "Investment"),
("Investing in startups offers the opportunity to be part of a vibrant community of investors, entrepreneurs, and industry experts.", "Investment"),
("Startups rely on investments to establish robust cybersecurity measures, safeguarding sensitive data and ensuring trust from customers.", "Investment"),
("Investments fuel the scaling efforts of startups, allowing them to increase production, meet growing demand, and achieve economies of scale.", "Investment"),
("Investing in startups fosters innovation, driving advancements in technology, healthcare, sustainability, and other critical sectors.", "Investment"),
("Startups utilize investments to conduct market research, validate product-market fit, and refine their value proposition to attract customers.", "Investment"),
("Investments enable startups to leverage partnerships and collaborations, accessing shared resources, expertise, and networks.", "Investment"),
("Investing in startups provides an avenue to support diverse founders and contribute to building an inclusive and equitable entrepreneurial ecosystem.", "Investment"),
("Startups rely on investments to establish robust supply chains, optimize logistics, and ensure efficient operations.", "Investment"),
("Investments fuel the expansion efforts of startups, empowering them to enter new verticals, diversify revenue streams, and capture untapped markets.", "Investment"),
("Mentors and advisors play a vital role in guiding startups through the complexities of entrepreneurship, offering invaluable insights and expertise.", "Mentors_and_advisors"),
("Startups greatly benefit from the wisdom and experience of mentors and advisors who can provide strategic direction and help navigate challenges.", "Mentors_and_advisors"),
("Having access to knowledgeable mentors and advisors gives startups a competitive advantage by accelerating their learning curve and avoiding common pitfalls.", "Mentors_and_advisors"),
("Mentors and advisors provide startups with a fresh perspective, challenging assumptions, and pushing them to think creatively and critically about their business.", "Mentors_and_advisors"),
("Startups often face uncertainties and ambiguity, and mentors and advisors provide much-needed clarity, helping them make informed decisions.", "Mentors_and_advisors"),
("Mentors and advisors act as a sounding board for startups, offering constructive feedback and helping them refine their ideas and strategies.", "Mentors_and_advisors"),
("The guidance and support of mentors and advisors can help startups build a strong foundation, establish best practices, and set realistic goals.", "Mentors_and_advisors"),
("Startups can tap into the networks of mentors and advisors, accessing valuable connections, partnerships, and opportunities for growth.", "Mentors_and_advisors"),
("Mentors and advisors bring a wealth of industry knowledge and insights, helping startups stay up-to-date with market trends and emerging technologies.", "Mentors_and_advisors"),
("For startups seeking investment, mentors and advisors can enhance their credibility and provide introductions to potential investors and funding sources.", "Mentors_and_advisors"),
("The mentorship and guidance of experienced professionals can instill confidence in startup founders, empowering them to overcome challenges and persevere.", "Mentors_and_advisors"),
("Mentors and advisors provide startups with a valuable support system, offering encouragement, motivation, and accountability along the entrepreneurial journey.", "Mentors_and_advisors"),
("Startups can leverage the expertise of mentors and advisors to build a strong team, attract top talent, and foster a culture of growth and innovation.", "Mentors_and_advisors"),
("Mentors and advisors act as trusted confidants for startup founders, providing a safe space to discuss ideas, concerns, and personal development.", "Mentors_and_advisors"),
("The mentorship of seasoned entrepreneurs can help startups anticipate and navigate common hurdles, accelerating their path to success.", "Mentors_and_advisors"),
("Mentors and advisors bring diverse perspectives to the table, challenging assumptions and helping startups think outside the box.", "Mentors_and_advisors"),
("Startups can learn from the successes and failures of mentors and advisors, gaining valuable lessons and avoiding costly mistakes.", "Mentors_and_advisors"),
("Mentors and advisors provide startups with access to a wealth of industry-specific knowledge, helping them make informed decisions and stay ahead of the curve.", "Mentors_and_advisors"),
("For startups entering new markets or industries, mentors and advisors with domain expertise can provide invaluable guidance and market insights.", "Mentors_and_advisors"),
("The mentorship and guidance of experienced professionals can help startups build resilience, adaptability, and a strong entrepreneurial mindset.", "Mentors_and_advisors"),
("Startups require mentors and advisors who have deep industry knowledge and experience to provide strategic guidance and insights.", "Mentors_and_advisors"),
("Mentors and advisors should have a strong track record of success in startups or entrepreneurship, demonstrating their ability to navigate challenges and drive growth.", "Mentors_and_advisors"),
("Startups need mentors and advisors who can challenge their assumptions, ask critical questions, and provide objective feedback to refine their business strategies.", "Mentors_and_advisors"),
("Mentors and advisors should possess excellent communication skills to effectively convey their expertise and guidance to startup founders and teams.", "Mentors_and_advisors"),
("Startups require mentors and advisors who are well-connected in the industry, able to open doors to valuable networks, partnerships, and funding opportunities.", "Mentors_and_advisors"),
("Mentors and advisors should have a genuine passion for startups and entrepreneurship, offering their support and guidance with a vested interest in the startup's success.", "Mentors_and_advisors"),
("Startups need mentors and advisors who are adaptable and flexible, able to navigate the rapidly changing business landscape and help startups stay ahead.", "Mentors_and_advisors"),
("Mentors and advisors should have a deep understanding of market trends, emerging technologies, and customer preferences to guide startups in making informed decisions.", "Mentors_and_advisors"),
("Startups require mentors and advisors who can provide constructive criticism, challenging startup founders to continuously improve and refine their business models.", "Mentors_and_advisors"),
("Mentors and advisors should be able to offer valuable insights and best practices in areas such as fundraising, marketing, operations, and talent acquisition.", "Mentors_and_advisors"),
("Startups need mentors and advisors who can help them identify and seize opportunities, guiding them in strategic partnerships, collaborations, and market expansions.", "Mentors_and_advisors"),
("Mentors and advisors should have a strong understanding of the startup's target market, customers, and competitive landscape to provide relevant guidance.", "Mentors_and_advisors"),
("Startups require mentors and advisors who can help them build a strong team, attract and retain top talent, and create a positive and inclusive work culture.", "Mentors_and_advisors"),
("Mentors and advisors should possess excellent problem-solving skills, helping startups navigate challenges, overcome obstacles, and make effective decisions.", "Mentors_and_advisors"),
("Startups need mentors and advisors who can provide guidance on legal and regulatory compliance, intellectual property protection, and risk management.", "Mentors_and_advisors"),
("Mentors and advisors should be supportive and empathetic, understanding the unique challenges and pressures faced by startup founders and teams.", "Mentors_and_advisors"),
("Startups require mentors and advisors who can help them develop a scalable business model, monetization strategies, and a path to sustainable growth and profitability.", "Mentors_and_advisors"),
("Mentors and advisors should have a growth mindset, encouraging startups to embrace experimentation, learn from failures, and iterate on their ideas and products.", "Mentors_and_advisors"),
("Startups need mentors and advisors who are committed and available, willing to invest time and effort in supporting the startup's growth and success.", "Mentors_and_advisors"),
("Mentors and advisors should possess strong leadership and mentorship skills, guiding startups in building resilience, fostering innovation, and developing their own leadership capabilities.", "Mentors_and_advisors"),
("In our startup, we are actively seeking mentors and advisors who can provide us with guidance and support.", "Mentors_and_advisors"),
("Startups should proactively seek feedback and insights from their target markets to continuously improve their products and customer experience.", "Target_Markets"),
("Identifying the influencers and key opinion leaders within their target markets can help startups amplify their brand reach and credibility.", "Target_Markets"),
("Startups need to adapt their pricing strategies to align with the purchasing power and willingness to pay of their target markets.", "Target_Markets"),
("Startups should conduct market segmentation to identify distinct groups within their target markets and tailor their strategies accordingly.", "Target_Markets"),
("Understanding the geographic location and distribution channels of their target markets helps startups optimize their logistics and distribution processes.", "Target_Markets"),
("Startups need to stay updated on the latest market trends and consumer behavior within their target markets to remain competitive.", "Target_Markets"),
("Identifying the pain points and unmet needs of their target markets allows startups to develop innovative solutions that resonate with customers.", "Target_Markets"),
("Startups should consider the psychographic factors such as values, interests, and lifestyles of their target markets when crafting their brand messaging.", "Target_Markets"),
("Understanding the purchasing behavior and decision-making process of their target markets helps startups design effective sales and marketing funnels.", "Target_Markets"),
("Startups should leverage social media and digital marketing channels to reach and engage with their target markets effectively.", "Target_Markets"),
("Identifying the competitive advantages and unique selling points that appeal to their target markets allows startups to differentiate themselves.", "Target_Markets"),
("Startups should conduct customer surveys and interviews to gather valuable insights and feedback from their target markets.", "Target_Markets")







]

# Split the examples into inputs (text) and labels (intents)
inputs = [example[0] for example in examples]
labels = [example[1] for example in examples]

# Preprocess the inputs
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(inputs)

def preprocess_text(text):
    # Remove special characters and extra whitespaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text)
    # Convert to lowercase
    text = text.lower()
    return text

st.title("Audio Recorder")
audio = audiorecorder("Click to record", "Recording...")

if len(audio) > 0:
    # To play audio in frontend:
    st.audio(audio.tobytes())

    # To save audio to a file:
    wav_file = open("audio.mp3", "wb")
    wav_file.write(audio.tobytes())
    wav_file.close()  # Close the file after writing

    # Load the audio data from the saved file
    audio_data = whisper.load_audio("audio.mp3")

    # Transcribe the audio
    model_w = whisper.load_model("base")
    result = model_w.transcribe(audio_data, fp16=False)
    text = result["text"]

    model = pickle.load(open('model.pkl', 'rb'))

    # Split the test paragraph into sentences
    sentences = re.split(r'\.\s*', text)

    # Iterate over each sentence and predict the intent
    with open('output.txt', 'a') as f:
        for sentence in sentences:
            # Remove sentences with less than 6 words
            if len(sentence.split()) < 6:
                continue

            preprocessed_sentence = preprocess_text(sentence)
            input_feature = vectorizer.transform([preprocessed_sentence])
            predicted_intent = model.predict(input_feature)[0]

            st.write("Sentence:", sentence)
            st.write("Predicted Intent:", predicted_intent)
            
            # Write the output to the file
            f.write("Line: " + sentence + "\n")
            f.write("Predicted Intent: " + predicted_intent + "\n")


    
    
    with open('output.txt') as f:
       st.download_button('Download CSV', f, 'text/csv') 
    with open('output.txt') as f:
       st.download_button('Download TXT', f) 







