class SectionBuilder:

    def __init__(self):
        pass

    # ---------------------------------------------------------

    def build(
        self,
        sections,
        paragraphs,
        parsed_tables,
    ):

        # -------------------------------------
        # Merge Pipeline
        # -------------------------------------

        self._assign_paragraphs_to_sections(
            sections,
            paragraphs
        )

        self._assign_tables_to_sections(
            sections,
            parsed_tables,
            paragraphs
        )

        self._calculate_section_page_span(
            sections
        )

        self._remove_duplicate_content(
            sections
        )

        self._validate_sections(
            sections
        )

        return sections

    # ---------------------------------------------------------

    def _get_last_paragraph(self, section):

        if section.children:
            child = section.children[-1]
            result = self._get_last_paragraph(child)
            if result is not None:
                return result

        if section.paragraphs:
            return section.paragraphs[-1]

        return None

    # ---------------------------------------------------------

    def _get_first_paragraph_or_heading(self, section):

        if section.paragraphs:
            return section.paragraphs[0]

        if section.children:
            return self._get_first_paragraph_or_heading(section.children[0])

        return None

    # ---------------------------------------------------------

    def _assign_paragraphs_to_sections(
        self,
        sections,
        paragraphs,
    ):

        if not sections:
            return

        self._assign_paragraphs_recursive(
            sections,
            paragraphs
        )

    # ---------------------------------------------------------

    def _assign_paragraphs_recursive(
        self,
        sections,
        paragraphs,
    ):

        for i, section in enumerate(sections):

            current_heading = section.title.strip()

            start = None

            # -------------------------
            # Find Current Heading
            # -------------------------

            for idx, para in enumerate(paragraphs):

                if (
                    para.block_type == "heading"
                    and
                    para.text.strip() == current_heading
                ):

                    start = idx + 1

                    break

            if start is None:

                if section.children:

                    self._assign_paragraphs_recursive(
                        section.children,
                        paragraphs
                    )

                continue

            # -------------------------
            # Find Next Heading
            # -------------------------

            end = len(paragraphs)

            if i + 1 < len(sections):

                next_heading = sections[i + 1].title.strip()

                for idx in range(start, len(paragraphs)):

                    para = paragraphs[idx]

                    if (
                        para.block_type == "heading"
                        and
                        para.text.strip() == next_heading
                    ):

                        end = idx

                        break

            # -------------------------
            # Assign Paragraph Objects
            # -------------------------

            section.paragraphs = []

            for para in paragraphs[start:end]:

                if para.block_type != "paragraph":
                    continue

                section.paragraphs.append(
                    para
                )

            # -------------------------
            # Recursive Children
            # -------------------------

            if section.children:

                self._assign_paragraphs_recursive(
                    section.children,
                    paragraphs
                )

    # ---------------------------------------------------------

    def _assign_tables_to_sections(
        self,
        sections,
        tables,
        paragraphs,
    ):

        if not sections or not tables:
            return

        self._assign_tables_recursive(
            sections,
            tables,
            paragraphs
        )

    # ---------------------------------------------------------

    def _assign_tables_recursive(
        self,
        sections,
        tables,
        paragraphs,
    ):

        for i, section in enumerate(sections):

            section.tables = []

            last_para = self._get_last_paragraph(section)
            first_para = self._get_first_paragraph_or_heading(section)

            # Spatial & Page Range Assignment
            for table in tables:

                # Case 1: Same page, table lies below last paragraph of the section
                if (
                    last_para is not None
                    and table.page_number == last_para.page_number
                    and table.top >= last_para.bottom
                ):
                    section.tables.append(table)

                # Case 2: Section spans across pages or table is right after heading
                elif (
                    first_para is not None
                    and section.page_start <= table.page_number <= section.page_end
                ):
                    section.tables.append(table)

            # -------------------------
            # Children
            # -------------------------

            if section.children:

                self._assign_tables_recursive(
                    section.children,
                    tables,
                    paragraphs
                )

    # ---------------------------------------------------------

    def _flatten_sections(
        self,
        sections,
        output,
    ):

        for section in sections:

            output.append(section)

            if section.children:

                self._flatten_sections(
                    section.children,
                    output
                )

    # ---------------------------------------------------------

    def _calculate_section_page_span(
        self,
        sections,
    ):

        for section in sections:

            pages = []

            # -----------------------
            # Paragraph Pages
            # -----------------------

            for para in section.paragraphs:

                pages.append(
                    para.page_number
                )

            # -----------------------
            # Table Pages
            # -----------------------

            for table in section.tables:

                pages.append(
                    table.page_number
                )

            # -----------------------
            # Child Sections
            # -----------------------

            if section.children:

                self._calculate_section_page_span(
                    section.children
                )

                for child in section.children:

                    if child.page_start:

                        pages.append(
                            child.page_start
                        )

                    if child.page_end:

                        pages.append(
                            child.page_end
                        )

            # -----------------------
            # Final Span
            # -----------------------

            if pages:

                section.page_start = min(pages)

                section.page_end = max(pages)

    # ---------------------------------------------------------

    def _remove_duplicate_content(
        self,
        sections,
    ):

        for section in sections:

            # -----------------------
            # Remove Duplicate Paragraphs
            # -----------------------

            unique_paragraphs = []

            seen = set()

            for para in section.paragraphs:

                if para.id not in seen:

                    unique_paragraphs.append(
                        para
                    )

                    seen.add(
                        para.id
                    )

            section.paragraphs = unique_paragraphs

            # -----------------------
            # Remove Duplicate Tables
            # -----------------------

            unique_tables = []

            seen = set()

            for table in section.tables:

                if table.table_id not in seen:

                    unique_tables.append(
                        table
                    )

                    seen.add(
                        table.table_id
                    )

            section.tables = unique_tables

            # -----------------------
            # Children
            # -----------------------

            if section.children:

                self._remove_duplicate_content(
                    section.children
                )

    # ---------------------------------------------------------

    def _validate_sections(
        self,
        sections,
    ):

        valid_sections = []

        for section in sections:

            # -----------------------
            # Validate Children First
            # -----------------------

            if section.children:

                self._validate_sections(
                    section.children
                )

            # -----------------------
            # Empty Section Check
            # -----------------------

            has_content = (

                len(section.paragraphs) > 0

                or

                len(section.tables) > 0

                or

                len(section.children) > 0

            )

            if has_content:

                valid_sections.append(
                    section
                )

        sections.clear()

        sections.extend(
            valid_sections
        )