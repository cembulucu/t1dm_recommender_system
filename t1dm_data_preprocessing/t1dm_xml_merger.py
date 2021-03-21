from xml.etree import ElementTree as eT


def merge_xmls(file1, file2, file_name):
    """ merge two xml files with same attributes """

    # get roots
    root1 = eT.parse(file1).getroot()
    root2 = eT.parse(file2).getroot()

    # for each attribute create a merged one and add it to the merged root
    merge_root = eT.Element('patient', attrib=root1.attrib)
    for r1_child, r2_child in zip(root1, root2):
        merged = r1_child
        merged.extend(r2_child)
        merge_root.append(merged)

    # create merged xml
    tree = eT.ElementTree(merge_root)
    tree.write(file_name, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    """ original dataset is split into training and test sets for offline learning, 
    merge two sets into one as online sequential learning is to be performed, without warm start """

    xml_root_path = '../t1dm_xmls/'
    xml_paths = ['559-ws-training.xml', '559-ws-testing.xml',
                 '563-ws-training.xml', '563-ws-testing.xml',
                 '570-ws-training.xml', '570-ws-testing.xml',
                 '575-ws-training.xml', '575-ws-testing.xml',
                 '588-ws-training.xml', '588-ws-testing.xml',
                 '591-ws-training.xml', '591-ws-testing.xml']

    xml_paths = [xml_root_path + p for p in xml_paths]
    merge_xmls(xml_paths[0], xml_paths[1], file_name=xml_root_path + 'patient_559_merged.xml')
    merge_xmls(xml_paths[2], xml_paths[3], file_name=xml_root_path + 'patient_563_merged.xml')
    merge_xmls(xml_paths[4], xml_paths[5], file_name=xml_root_path + 'patient_570_merged.xml')
    merge_xmls(xml_paths[6], xml_paths[7], file_name=xml_root_path + 'patient_575_merged.xml')
    merge_xmls(xml_paths[8], xml_paths[9], file_name=xml_root_path + 'patient_588_merged.xml')
    merge_xmls(xml_paths[10], xml_paths[11], file_name=xml_root_path + 'patient_591_merged.xml')
